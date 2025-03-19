import os
import re
import nltk
import shutil
import logging
import time  # Add import for time
from typing import List, Dict, Union, Tuple, Optional
from bs4 import BeautifulSoup
import requests
from PyPDF2 import PdfReader
import chromadb
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("document_processor")

class DocumentProcessor:
    def __init__(
        self, 
        pdf_dir: str,
        url_sources: List[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 512,  # Default to 512 dimensions
        chroma_dir: str = "./chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        chunk_method: str = "hybrid",  # Parameter to control chunking method
        clear_chroma: bool = True  # New parameter to control whether to clear the Chroma DB
    ):
        self.pdf_dir = pdf_dir
        self.url_sources = url_sources or []
        self.chroma_dir = chroma_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_method = chunk_method
        self.clear_chroma = clear_chroma  # Store the parameter
        
        # Always use fixed 512 embedding dimensions regardless of input
        self.embedding_dim = 512
        self.documents = []
        
        # Initialize embedding model directly with fixed 512 dimensions
        logger.info(f"Initializing embedding model: {embedding_model} with fixed 512 dimensions")
        self.embed_model = self._initialize_embedding_model(embedding_model)
        
        # Download NLTK resources properly
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)  # For better text analysis
            logger.info("NLTK resources downloaded successfully")
        except Exception as e:
            logger.warning(f"Could not download NLTK resources: {e}")
    
    def _initialize_embedding_model(self, model_name: str):
        """Initialize embedding model without dimension reduction, fixed at 512 dimensions"""
        try:
            # Standard initialization without dimension reduction
            logger.info(f"Initializing standard embedding model: {model_name}")
            return HuggingFaceEmbedding(
                model_name=model_name,
                embed_batch_size=4  # Reasonable batch size to avoid memory issues
            )
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def load_documents(self) -> List[Document]:
        """Load all documents from specified directories and URLs"""
        logger.info(f"Loading documents from {self.pdf_dir}...")
        
        # Process all PDFs in the directory and subdirectories
        for root, _, files in os.walk(self.pdf_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    category = os.path.basename(os.path.dirname(file_path))
                    self.load_pdf(file_path, category)
        
        # Load from URLs if provided
        for url in self.url_sources:
            self.load_from_url(url)
            
        logger.info(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def load_pdf(self, file_path: str, category: str = "Unknown") -> None:
        """Load a single PDF document"""
        try:
            reader = PdfReader(file_path)
            text_chunks = []
            
            # Extract text from each page
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        # Include page number in text for better traceability
                        page_prefix = f"Page {page_num+1}: "
                        text_chunks.append(page_prefix + text)
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num} in {file_path}: {str(e)}")
            
            if not text_chunks:
                logger.warning(f"No text extracted from {file_path}")
                return
            
            # Clean text before creating document
            doc_text = "\n\n".join(text_chunks)
            doc_text = self.clean_text(doc_text)
            
            # Create document with useful metadata
            doc = Document(
                text=doc_text,
                metadata={
                    "source": file_path,
                    "file_name": os.path.basename(file_path),
                    "category": category,
                    "type": "pdf",
                    "pages": len(reader.pages),
                    "title": self._extract_title(doc_text, os.path.basename(file_path))
                }
            )
            self.documents.append(doc)
            logger.info(f"Loaded PDF: {os.path.basename(file_path)} ({len(doc_text)} chars, {len(reader.pages)} pages)")
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
    
    def _extract_title(self, text: str, default_name: str) -> str:
        """Extract a title from the document text"""
        # Try to find a title in the first 1000 characters
        first_chunk = text[:1000]
        
        # Look for patterns that might indicate titles
        title_patterns = [
            r'Title:\s*([^\n]+)',
            r'^\s*([A-Z][^.!?]{10,100})[\n\.]',  # Capitalized first line
            r'^\s*#\s+([^\n]+)',  # Markdown style title
            r'^\s*([A-Z][A-Z\s]{10,100}[A-Z])\s*$',  # ALL CAPS TITLE
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, first_chunk, re.MULTILINE)
            if match:
                candidate = match.group(1).strip()
                # Filter out very long or very short candidates
                if 3 < len(candidate) < 100:
                    return candidate
        
        # If no good title found, use filename as fallback
        return default_name
    
    def load_from_url(self, url: str) -> None:
        """Load content from a URL"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove non-content elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'form']):
                element.decompose()
            
            # Get page title
            page_title = soup.title.string if soup.title else "Untitled Page"
            
            # Extract text - prioritize article content
            article_content = soup.find('article')
            if article_content:
                # If there's an article element, focus on that
                paragraphs = article_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'li'])
                content = "\n\n".join([el.get_text(strip=True) for el in paragraphs if el.get_text(strip=True)])
            else:
                # Otherwise get text from all content elements
                content_elements = soup.find_all(['p', 'article', 'section', 'h1', 'h2', 'h3', 'div.content', 'main'])
                content = "\n\n".join([element.get_text(strip=True) for element in content_elements])
            
            if not content.strip():
                logger.warning(f"No content extracted from {url}")
                return
            
            # Clean text before creating document
            text = self.clean_text(content)
            
            # Create document
            doc = Document(
                text=text,
                metadata={
                    "source": url,
                    "type": "url",
                    "title": page_title,
                    "category": "Web"
                }
            )
            self.documents.append(doc)
            logger.info(f"Loaded URL: {url} ({len(text)} chars)")
            
        except Exception as e:
            logger.error(f"Error loading URL {url}: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean text to avoid tokenization issues"""
        # Replace null bytes and other control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Convert multiple newlines to double newline (preserves paragraph structure)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Normalize whitespace within lines
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove extremely long words that might cause issues
        text = re.sub(r'\S{1000,}', '[LONG_TEXT]', text)
        
        # Remove any potentially problematic characters
        text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
        
        return text.strip()
    
    def _safe_sent_tokenize(self, text: str) -> List[str]:
        """Safely tokenize text into sentences with fallback"""
        try:
            return nltk.sent_tokenize(text)
        except:
            # Fallback to simple regex approach
            logger.warning("NLTK tokenization failed, using simple regex fallback")
            return re.split(r'(?<=[.!?])\s+', text)
    
    def process_documents(self) -> List[Document]:
        """
        Process documents using document-type aware chunking
        
        This enhanced method detects document types and applies specialized chunking strategies
        """
        if not self.documents:
            logger.warning("No documents to process. Run load_documents() first.")
            return []
        
        chunked_docs = []
        total_chunks_created = 0
        
        for doc in self.documents:
            try:
                # Detect document type
                doc_type = self._detect_document_type(doc)
                logger.info(f"Document '{doc.metadata.get('title', 'Untitled')}' detected as: {doc_type}")
                
                # Apply appropriate chunking strategy based on document type
                if doc_type == "textbook":
                    # Use chapter-based chunking for textbooks
                    doc_chunks = self.apply_chapter_chunking(doc)
                elif doc_type == "technical_paper":
                    # Use section-based chunking for technical papers
                    doc_chunks = self.apply_section_chunking(doc)
                else:
                    # Default to hybrid chunking for unknown document types
                    # Create a list to hold chunks just for this document
                    doc_chunks = []
                    
                    # STEP 1: Extract sections by headings
                    sections = self._split_by_section_headings(doc.text)
                    
                    logger.info(f"Document '{doc.metadata.get('title', 'Untitled')}': Found {len(sections)} sections")
                    
                    # STEP 2: Process each section
                    for i, section_text in enumerate(sections):
                        # Skip empty sections
                        if not section_text.strip():
                            continue
                        
                        # Extract section title
                        section_title = self._extract_section_title(section_text)
                        logger.debug(f"Processing section: {section_title}")
                        
                        # Split by paragraphs
                        paragraphs = re.split(r'\n\s*\n', section_text)
                        
                        # Fixed size context window with paragraph boundaries
                        current_chunk = ""
                        section_chunks = []
                        
                        for para in paragraphs:
                            para_trimmed = para.strip()
                            if not para_trimmed:
                                continue
                                
                            # If adding this paragraph would exceed the chunk size
                            if len(current_chunk) + len(para_trimmed) > self.chunk_size:
                                # Save current chunk if not empty
                                if current_chunk.strip():
                                    section_chunks.append(current_chunk.strip())
                                
                                # Start a new chunk, possibly with overlap
                                if self.chunk_overlap > 0 and current_chunk:
                                    # Try to create overlap at sentence boundaries
                                    sentences = self._safe_sent_tokenize(current_chunk)
                                    overlap_text = ""
                                    
                                    # Build overlap from the last few sentences
                                    for sent in reversed(sentences):
                                        if len(overlap_text) + len(sent) <= self.chunk_overlap:
                                            overlap_text = sent + " " + overlap_text
                                        else:
                                            break
                                    
                                    current_chunk = overlap_text.strip()
                                else:
                                    current_chunk = ""
                            
                            # Add paragraph to current chunk
                            if current_chunk and not current_chunk.endswith('\n'):
                                current_chunk += "\n\n"
                            current_chunk += para_trimmed
                        
                        # Add the last chunk if not empty
                        if current_chunk.strip():
                            section_chunks.append(current_chunk.strip())
                        
                        # If section would create too many small chunks, try to recombine some
                        if len(section_chunks) > 0 and all(len(chunk) < self.chunk_size // 2 for chunk in section_chunks):
                            optimized_chunks = self._optimize_small_chunks(section_chunks)
                            section_chunks = optimized_chunks
                        
                        # Create documents from section chunks
                        for j, chunk_text in enumerate(section_chunks):
                            doc_chunks.append(Document(
                                text=chunk_text,
                                metadata={
                                    **doc.metadata,
                                    "chunk_type": "section",
                                    "section_index": i,
                                    "chunk_index": j,
                                    "section_title": section_title,
                                    "section_count": len(sections),
                                    "is_table_of_contents": "content" in section_title.lower() and i == 0
                                }
                            ))
                
                # If we didn't get enough chunks, fall back to more aggressive chunking
                if len(doc_chunks) <= 1:
                    logger.info(f"Document produced only {len(doc_chunks)} chunks, trying sentence chunking")
                    
                    # Try sentence-based chunking as fallback
                    sentence_chunks = self._chunk_by_sentences(doc.text)
                    
                    if len(sentence_chunks) > len(doc_chunks):
                        logger.info(f"Switched to sentence chunking: {len(sentence_chunks)} chunks")
                        doc_chunks = [
                            Document(
                                text=chunk,
                                metadata={
                                    **doc.metadata,
                                    "chunk_type": "sentence_fallback", 
                                    "chunk_index": i
                                }
                            ) for i, chunk in enumerate(sentence_chunks)
                        ]
                
                chunked_docs.extend(doc_chunks)
                total_chunks_created += len(doc_chunks)
                logger.info(f"Created {len(doc_chunks)} chunks from document '{doc.metadata.get('title', 'Untitled')}'")
                
            except Exception as e:
                logger.error(f"Error chunking document {doc.metadata.get('file_name', 'unknown')}: {str(e)}")
                chunked_docs.extend(self._apply_fallback_chunking(doc))
                total_chunks_created += len(self._apply_fallback_chunking(doc))
        
        logger.info(f"Total chunks created: {total_chunks_created} from {len(self.documents)} documents")
        
        if total_chunks_created <= len(self.documents):
            logger.warning("Warning: Number of chunks <= number of documents. Chunking may not be effective.")
            
        return chunked_docs
    
    def _detect_document_type(self, doc: Document) -> str:
        """
        Detect document type based on content and structure
        
        Args:
            doc: Document to analyze
            
        Returns:
            Document type: "textbook", "technical_paper", or "general"
        """
        # Extract metadata and text for analysis
        metadata = doc.metadata
        text = doc.text.lower()
        file_name = metadata.get('file_name', '').lower()
        page_count = metadata.get('pages', 0)
        
        # Check for textbook indicators
        textbook_indicators = 0
        
        # Check filename for textbook indicators
        if any(kw in file_name for kw in ['book', 'textbook', 'guide', 'dummies']):
            textbook_indicators += 2
        
        # Check for chapter patterns in text
        if re.search(r'chapter\s+\d+', text):
            textbook_indicators += 3
            
        if re.search(r'table of contents', text):
            textbook_indicators += 1
            
        # Check for consecutive chapter headings
        if re.search(r'chapter\s+\d+.*?chapter\s+\d+', text, re.DOTALL):
            textbook_indicators += 2
            
        # Check for long documents (textbooks tend to be longer)
        if page_count > 100:
            textbook_indicators += 1
            
        # Check for technical paper indicators
        paper_indicators = 0
        
        # Technical papers usually have these sections
        if re.search(r'abstract', text) and re.search(r'introduction', text):
            paper_indicators += 2
            
        if re.search(r'conclusion', text) and re.search(r'references', text):
            paper_indicators += 2
            
        # Technical papers often contain these terms
        if re.search(r'et al\.', text) or re.search(r'cited', text):
            paper_indicators += 1
            
        # Papers typically have .pdf extension and are shorter
        if file_name.endswith('.pdf') and page_count < 50:
            paper_indicators += 1
            
        # Check for arxiv-style papers
        if re.search(r'\d{4}\.\d{5}', file_name):  # arxiv ID pattern
            paper_indicators += 3
            
        # Evaluate and return document type
        if textbook_indicators >= 3 and textbook_indicators > paper_indicators:
            return "textbook"
        elif paper_indicators >= 3 and paper_indicators > textbook_indicators:
            return "technical_paper"
        else:
            return "general"
    
    def apply_chapter_chunking(self, doc: Document) -> List[Document]:
        """
        Apply chapter-based chunking for textbooks with performance optimizations
        
        Args:
            doc: Document to chunk by chapters
            
        Returns:
            List of chunked documents
        """
        logger.info(f"Applying chapter-based chunking to '{doc.metadata.get('title', 'Untitled')}'")
        
        chunked_docs = []
        text = doc.text
        
        # Limit processing time for very large documents
        if len(text) > 5000000:  # If > 5MB, simplify approach
            logger.warning(f"Document is very large ({len(text)} chars). Using simplified chapter detection.")
            # Quick scan for chapter patterns without expensive regex
            has_chapters = "chapter 1" in text.lower() or "chapter i" in text.lower()
            if not has_chapters:
                logger.info("No obvious chapter structure detected. Using section-based chunking.")
                return self.apply_section_chunking(doc)
        
        # Find chapter boundaries using multiple patterns - optimized for speed
        chapter_patterns = [
            # Most common chapter formats first for early success
            r'\n\s*Chapter\s+\d+[\s:]+([^\n]{3,100})',  # Standard "Chapter N: Title" format
            r'\n\s*CHAPTER\s+\d+[\s:]+([^\n]{3,100})',  # All-caps "CHAPTER N: Title"
            r'\n\s*\d+\.\s+([A-Z][^\n]{3,100})',  # Numbered sections like "1. INTRODUCTION"
            r'\n\s*Part\s+\d+[\s:]+([^\n]{3,100})'  # Parts instead of chapters
        ]
        
        # Find all chapter headings - with timeout protection
        start_time = time.time()
        chapters = []
        pattern_search_timeout = 30  # seconds
        
        for pattern in chapter_patterns:
            # Check if we're taking too long
            if time.time() - start_time > pattern_search_timeout:
                logger.warning(f"Chapter pattern search taking too long (>{pattern_search_timeout}s). Using fallback method.")
                return self.apply_section_chunking(doc)
                
            # Use non-greedy matching and limit search complexity
            matches = list(re.finditer(pattern, text[:min(len(text), 500000)], re.IGNORECASE | re.MULTILINE))
            
            if matches and len(matches) > 1:  # Need at least 2 chapters
                logger.info(f"Found {len(matches)} chapters using pattern: {pattern[:30]}...")
                
                # Get first few and last few matches to avoid processing too many
                if len(matches) > 50:  # Limit to 50 chapters max
                    logger.warning(f"Too many chapter matches ({len(matches)}). Limiting to 50 chapters.")
                    matches = matches[:50]
                
                # Record chapter positions
                for i, match in enumerate(matches):
                    start_pos = match.start()
                    # End position is the start of the next chapter or the end of text
                    end_pos = matches[i+1].start() if i < len(matches)-1 else len(text)
                    chapter_title = match.group(1).strip() if match.groups() else f"Chapter {i+1}"
                    
                    chapters.append({
                        "title": chapter_title,
                        "start": start_pos,
                        "end": end_pos
                    })
                
                # If we found chapters with this pattern, stop looking
                if len(chapters) > 1:
                    break
        
        # If no chapters found or search took too long, try splitting by section headers
        if len(chapters) < 2 or (time.time() - start_time > pattern_search_timeout):
            logger.info(f"Few or no chapters found by patterns. Using section-based chunking.")
            return self.apply_section_chunking(doc)
        
        # Process each chapter - with progress logging and timeout protection
        chapter_processing_timeout = 120  # seconds
        chapter_processing_start = time.time()
        
        for i, chapter in enumerate(chapters):
            # Check if we're taking too long overall
            if time.time() - chapter_processing_start > chapter_processing_timeout:
                logger.warning(f"Chapter processing taking too long (>{chapter_processing_timeout}s). "
                              f"Processed {i}/{len(chapters)} chapters. Returning current chunks.")
                # Return what we have so far
                if chunked_docs:
                    logger.info(f"Returning {len(chunked_docs)} chunks from partial chapter processing")
                    return chunked_docs
                else:
                    logger.warning("No chunks created yet. Falling back to section chunking.")
                    return self.apply_section_chunking(doc)
            
            # Log progress
            if i % 5 == 0 or i == len(chapters) - 1:
                logger.info(f"Processing chapter {i+1}/{len(chapters)}")
                
            # Extract chapter text
            try:
                if chapter["start"] < 0 or chapter["end"] > len(text) or chapter["start"] >= chapter["end"]:
                    logger.warning(f"Invalid chapter boundaries: {chapter['start']}:{chapter['end']} (text length: {len(text)})")
                    continue
                    
                chapter_text = text[chapter["start"]:chapter["end"]]
                chapter_title = chapter["title"]
                
                # Skip very short chapters (likely false positives)
                if len(chapter_text.split()) < 50:
                    continue
                    
                # Skip unreasonably large chapters (likely parsing errors)
                if len(chapter_text) > 500000:  # > 500KB
                    logger.warning(f"Chapter {i+1} is too large ({len(chapter_text)} chars). Creating smaller chunks.")
                    # Directly split into fixed chunks for very large chapters
                    fixed_chunks = self._chunk_by_fixed_size(chapter_text)
                    for k, chunk_text in enumerate(fixed_chunks):
                        chunked_docs.append(Document(
                            text=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk_type": "chapter_fixed",
                                "chapter_index": i,
                                "chapter_title": chapter_title,
                                "chunk_index": k
                            }
                        ))
                    continue
                
                # Handle regular chapters
                if len(chapter_text) > self.chunk_size:
                    # Process in sections, but limit number of sections
                    sections = self._split_by_section_headings(chapter_text)
                    logger.info(f"Chapter {i+1} has {len(sections)} sections")
                    
                    # Limit number of sections to process
                    max_sections = 30
                    if len(sections) > max_sections:
                        logger.warning(f"Too many sections ({len(sections)}) in chapter {i+1}. Limiting to {max_sections}.")
                        sections = sections[:max_sections]
                    
                    for j, section_text in enumerate(sections):
                        # Skip empty sections
                        if not section_text.strip():
                            continue
                            
                        try:
                            # Keep section processing simple
                            if len(section_text) > self.chunk_size:
                                # Split into paragraph chunks
                                paragraphs = re.split(r'\n\s*\n', section_text)
                                current_chunk = ""
                                section_chunks = []
                                
                                for para in paragraphs:
                                    para = para.strip()
                                    if not para:
                                        continue
                                    
                                    if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                                        if current_chunk:
                                            current_chunk += "\n\n"
                                        current_chunk += para
                                    else:
                                        if current_chunk:
                                            section_chunks.append(current_chunk)
                                        current_chunk = para
                                        
                                if current_chunk:
                                    section_chunks.append(current_chunk)
                                    
                                # Create documents from chunks
                                for k, chunk_text in enumerate(section_chunks):
                                    chunked_docs.append(Document(
                                        text=chunk_text,
                                        metadata={
                                            **doc.metadata,
                                            "chunk_type": "chapter_section",
                                            "chapter_index": i,
                                            "chapter_title": chapter_title,
                                            "section_index": j,
                                            "chunk_index": k
                                        }
                                    ))
                            else:
                                # Use section as is
                                chunked_docs.append(Document(
                                    text=section_text,
                                    metadata={
                                        **doc.metadata,
                                        "chunk_type": "chapter_section",
                                        "chapter_index": i,
                                        "chapter_title": chapter_title,
                                        "section_index": j
                                    }
                                ))
                        except Exception as e:
                            logger.error(f"Error processing section {j} of chapter {i+1}: {str(e)}")
                            # Continue with next section
                else:
                    # Chapter fits in chunk size, use it as is
                    chunked_docs.append(Document(
                        text=chapter_text,
                        metadata={
                            **doc.metadata,
                            "chunk_type": "chapter",
                            "chapter_index": i,
                            "chapter_title": chapter_title
                        }
                    ))
            except Exception as e:
                logger.error(f"Error processing chapter {i+1}: {str(e)}")
                # Continue with next chapter
        
        # If we didn't get chunks, fall back to hybrid chunking
        if not chunked_docs:
            logger.warning(f"Chapter chunking failed to produce chunks. Falling back to hybrid chunking.")
            return self.apply_hybrid_chunking(doc)
        
        logger.info(f"Created {len(chunked_docs)} chunks from chapter-based chunking")
        return chunked_docs
    
    def apply_section_chunking(self, doc: Document) -> List[Document]:
        """
        Apply section-based chunking for technical papers and reports
        
        Args:
            doc: Document to chunk by sections
            
        Returns:
            List of chunked documents
        """
        logger.info(f"Applying section-based chunking to '{doc.metadata.get('title', 'Untitled')}'")
        
        chunked_docs = []
        text = doc.text
        
        # Extract sections using enhanced section detection
        sections = self._split_by_section_headings(text)
        
        # Process each section
        for i, section_text in enumerate(sections):
            # Skip empty sections
            if not section_text.strip():
                continue
                
            # Extract section title
            section_title = self._extract_section_title(section_text)
            logger.debug(f"Processing section: {section_title}")
            
            # Check if the section is too large
            if len(section_text) > self.chunk_size:
                # Split large sections by paragraphs
                paragraphs = re.split(r'\n\s*\n', section_text)
                
                # Create chunks from paragraphs
                current_chunk = ""
                chunks = []
                
                for para in paragraphs:
                    para_trimmed = para.strip()
                    if not para_trimmed:
                        continue
                        
                    # If adding this paragraph would exceed the chunk size
                    if len(current_chunk) + len(para_trimmed) > self.chunk_size:
                        # Save current chunk if not empty
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        
                        # Start a new chunk
                        current_chunk = para_trimmed
                    else:
                        # Add paragraph to current chunk
                        if current_chunk and not current_chunk.endswith('\n'):
                            current_chunk += "\n\n"
                        current_chunk += para_trimmed
                
                # Add the last chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Create documents from chunks
                for j, chunk_text in enumerate(chunks):
                    chunked_docs.append(Document(
                        text=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_type": "paper_section",
                            "section_index": i,
                            "section_title": section_title,
                            "chunk_index": j,
                            "section_count": len(sections)
                        }
                    ))
            else:
                # Section fits in a chunk, use it as is
                chunked_docs.append(Document(
                    text=section_text,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "paper_section",
                        "section_index": i,
                        "section_title": section_title,
                        "section_count": len(sections)
                    }
                ))
        
        # If we didn't get chunks, fall back to sentence chunking
        if not chunked_docs:
            logger.warning(f"Section chunking failed to produce chunks. Falling back to sentence chunking.")
            
            # Fallback to sentence-based chunking
            sentence_chunks = self._chunk_by_sentences(text)
            
            chunked_docs = [
                Document(
                    text=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "sentence_fallback", 
                        "chunk_index": i
                    }
                ) for i, chunk in enumerate(sentence_chunks)
            ]
        
        return chunked_docs
    
    def apply_hybrid_chunking(self, doc: Document = None) -> List[Document]:
        """
        Apply a hybrid chunking strategy optimized for financial documents
        
        This combines semantic, recursive, and section-based approaches
        
        Args:
            doc: Optional single document to chunk (if not provided, processes all documents)
            
        Returns:
            List of Document objects after chunking
        """
        if doc is not None:
            # Process a single document
            documents_to_process = [doc]
        else:
            # Process all documents
            documents_to_process = self.documents
            
            if not documents_to_process:
                logger.warning("No documents to chunk. Run load_documents() first.")
                return []
        
        chunked_docs = []
        total_chunks_created = 0
        
        for doc in documents_to_process:
            try:
                # STEP 1: Extract sections by headings
                sections = self._split_by_section_headings(doc.text)
                doc_chunks = []
                
                logger.info(f"Document '{doc.metadata.get('title', 'Untitled')}': Found {len(sections)} sections")
                
                # STEP 2: Process each section
                for i, section_text in enumerate(sections):
                    # Skip empty sections
                    if not section_text.strip():
                        continue
                    
                    # Extract section title
                    section_title = self._extract_section_title(section_text)
                    logger.debug(f"Processing section: {section_title}")
                    
                    # Split by paragraphs
                    paragraphs = re.split(r'\n\s*\n', section_text)
                    
                    # Fixed size context window with paragraph boundaries
                    current_chunk = ""
                    section_chunks = []
                    
                    for para in paragraphs:
                        para_trimmed = para.strip()
                        if not para_trimmed:
                            continue
                            
                        # If adding this paragraph would exceed the chunk size
                        if len(current_chunk) + len(para_trimmed) > self.chunk_size:
                            # Save current chunk if not empty
                            if current_chunk.strip():
                                section_chunks.append(current_chunk.strip())
                            
                            # Start a new chunk, possibly with overlap
                            if self.chunk_overlap > 0 and current_chunk:
                                # Try to create overlap at sentence boundaries
                                sentences = self._safe_sent_tokenize(current_chunk)
                                overlap_text = ""
                                
                                # Build overlap from the last few sentences
                                for sent in reversed(sentences):
                                    if len(overlap_text) + len(sent) <= self.chunk_overlap:
                                        overlap_text = sent + " " + overlap_text
                                    else:
                                        break
                                
                                current_chunk = overlap_text.strip()
                            else:
                                current_chunk = ""
                        
                        # Add paragraph to current chunk
                        if current_chunk and not current_chunk.endswith('\n'):
                            current_chunk += "\n\n"
                        current_chunk += para_trimmed
                    
                    # Add the last chunk if not empty
                    if current_chunk.strip():
                        section_chunks.append(current_chunk.strip())
                    
                    # If section would create too many small chunks, try to recombine some
                    if len(section_chunks) > 0 and all(len(chunk) < self.chunk_size // 2 for chunk in section_chunks):
                        optimized_chunks = self._optimize_small_chunks(section_chunks)
                        section_chunks = optimized_chunks
                    
                    # Create documents from section chunks
                    for j, chunk_text in enumerate(section_chunks):
                        doc_chunks.append(Document(
                            text=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk_type": "section",
                                "section_index": i,
                                "chunk_index": j,
                                "section_title": section_title,
                                "section_count": len(sections),
                                "is_table_of_contents": "content" in section_title.lower() and i == 0
                            }
                        ))
                
                # If we didn't get enough chunks, fall back to more aggressive chunking
                if len(doc_chunks) <= 1:
                    logger.info(f"Document produced only {len(doc_chunks)} chunks, trying sentence chunking")
                    
                    # Try sentence-based chunking as fallback
                    sentence_chunks = self._chunk_by_sentences(doc.text)
                    
                    if len(sentence_chunks) > len(doc_chunks):
                        logger.info(f"Switched to sentence chunking: {len(sentence_chunks)} chunks")
                        doc_chunks = [
                            Document(
                                text=chunk,
                                metadata={
                                    **doc.metadata,
                                    "chunk_type": "sentence_fallback", 
                                    "chunk_index": i
                                }
                            ) for i, chunk in enumerate(sentence_chunks)
                        ]
                    
                chunked_docs.extend(doc_chunks)
                total_chunks_created += len(doc_chunks)
                logger.info(f"Created {len(doc_chunks)} chunks from document '{doc.metadata.get('title', 'Untitled')}'")
                
            except Exception as e:
                logger.error(f"Error chunking document {doc.metadata.get('file_name', 'unknown')}: {str(e)}")
                fallback_docs = self._apply_fallback_chunking(doc)
                chunked_docs.extend(fallback_docs)
                total_chunks_created += len(fallback_docs)
        
        if doc is None:  # Only log this for multi-document processing
            logger.info(f"Total chunks created: {total_chunks_created} from {len(self.documents)} documents")
            
            if total_chunks_created <= len(self.documents):
                logger.warning("Warning: Number of chunks <= number of documents. Chunking may not be effective.")
            
        return chunked_docs
    
    def _apply_fallback_chunking(self, doc: Document) -> List[Document]:
        """Apply fallback chunking strategies when other methods fail"""
        logger.warning(f"Attempting fallback chunking for document: {doc.metadata.get('file_name', 'unknown')}")
        
        try:
            # Sentence-based chunking
            sentence_chunks = self._chunk_by_sentences(doc.text)
            fallback_docs = [
                Document(
                    text=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "sentence_fallback", 
                        "chunk_index": i
                    }
                ) for i, chunk in enumerate(sentence_chunks)
            ]
            
            logger.info(f"Created {len(fallback_docs)} fallback chunks from document (sentence-based)")
            return fallback_docs
            
        except Exception as e2:
            logger.error(f"Sentence-based fallback chunking also failed: {str(e2)}")
            
            # Last resort: fixed-size chunks
            try:
                fixed_chunks = self._chunk_by_fixed_size(doc.text)
                emergency_docs = [
                    Document(
                        text=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_type": "fixed_fallback", 
                            "chunk_index": i
                        }
                    ) for i, chunk in enumerate(fixed_chunks)
                ]
                
                logger.info(f"Created {len(emergency_docs)} emergency chunks (fixed-size)")
                return emergency_docs
                
            except Exception as e3:
                logger.error(f"All chunking methods failed: {str(e3)}")
                
                # Absolute last resort: single chunk with truncated text
                return [Document(
                    text=doc.text[:self.chunk_size],
                    metadata={
                        **doc.metadata,
                        "chunk_type": "truncated_emergency",
                        "truncated": True
                    }
                )]
    
    def _optimize_small_chunks(self, chunks: List[str]) -> List[str]:
        """Optimize small chunks by combining them when possible"""
        if not chunks:
            return []
            
        result = []
        current = chunks[0]
        
        for i in range(1, len(chunks)):
            if len(current) + len(chunks[i]) + 2 <= self.chunk_size:
                # Can combine
                current += "\n\n" + chunks[i]
            else:
                # Can't combine - store current and start a new one
                result.append(current)
                current = chunks[i]
                
        # Don't forget the last chunk
        if current:
            result.append(current)
            
        return result
    
    def _split_by_section_headings(self, text: str) -> List[str]:
        """
        Split text by section headings with improved patterns for financial documents
        
        Args:
            text: Input text to split
            
        Returns:
            List of section texts
        """
        # Performance optimization for very large texts
        text_to_analyze = text
        if len(text) > 1000000:  # If > 1MB, only analyze first part
            logger.warning(f"Text is very large ({len(text)} chars). Using first 500KB for section detection.")
            # Keep the full text for actual sections, but limit what we analyze for headings
            text_to_analyze = text[:500000]  # Use first 500KB for pattern detection
        
        # Match common section heading patterns - simplified for speed
        heading_patterns = [
            # Numbered headings (most common first)
            r'\n\s*\d+\.\s+[A-Z]',  # 1. TITLE (most common)
            r'\n\s*\d+\.\d+\s+[A-Z]',  # 1.1 TITLE
            r'\n\s*[A-Z][A-Z\s]{5,}\n',  # ALL CAPS HEADING
            r'\n\s*(?:SECTION|Section)\s+\d+',  # Section 1
            r'\n\s*(?:CHAPTER|Chapter)\s+\d+',  # Chapter 1
            
            # Markdown style headings
            r'\n\s*#{1,3}\s+\w',  # # Heading (only first 3 levels, must have content)
            
            # Common financial doc headings (simplified)
            r'\n\s*EXECUTIVE\s+SUMMARY',
            r'\n\s*INTRODUCTION\b',
            r'\n\s*BACKGROUND\b',
            r'\n\s*CONCLUSION\b',
            r'\n\s*APPENDIX\b',
            
            # Roman numerals (simplified)
            r'\n\s*[IVX]+\.\s+[A-Z]',  # I. TITLE, limited to first 30
        ]
        
        # Add technical paper section patterns (simplified)
        paper_heading_patterns = [
            r'\n\s*Abstract\b',
            r'\n\s*Introduction\b',
            r'\n\s*Related\s+Work\b',
            r'\n\s*Method(?:ology)?\b',
            r'\n\s*Experiment',
            r'\n\s*Results\b',
            r'\n\s*Discussion\b',
            r'\n\s*Conclusion\b',
            r'\n\s*References\b'
        ]
        
        # Use timeout protection
        start_time = time.time()
        pattern_search_timeout = 20  # seconds
        
        try:
            # Use a simple approach first - try to find patterns individually
            # This is faster than a combined regex for large documents
            matches = []
            
            # Record all section boundary positions
            for pattern in heading_patterns + paper_heading_patterns:
                # Check if we're taking too long
                if time.time() - start_time > pattern_search_timeout:
                    logger.warning(f"Section pattern search taking too long. Using simplified approach.")
                    break
                    
                # Find matches for this pattern
                try:
                    pattern_matches = list(re.finditer(pattern, text_to_analyze, re.IGNORECASE | re.MULTILINE))
                    matches.extend(pattern_matches)
                except Exception as e:
                    logger.warning(f"Error with pattern '{pattern[:20]}...': {e}")
                    continue
            
            # Sort matches by position
            matches.sort(key=lambda m: m.start())
            
            # Limit number of sections for performance
            max_sections = 100
            if len(matches) > max_sections:
                logger.warning(f"Too many section matches ({len(matches)}). Limiting to {max_sections} sections.")
                matches = matches[:max_sections]
                
            # Check if we have a reasonable number of sections
            if len(matches) >= 2:
                # Split text at section boundaries
                sections = []
                
                # Add introduction if present
                if matches[0].start() > 0:
                    intro_text = text[:matches[0].start()].strip()
                    if intro_text and len(intro_text) > 100:  # Only add substantial intros
                        sections.append(intro_text)
                
                # Process each section
                for i, match in enumerate(matches):
                    start = match.start()
                    # Find the end of this section (start of next or end of text)
                    end = matches[i+1].start() if i < len(matches) - 1 else len(text)
                    
                    # Skip sections that appear invalid
                    if end <= start or end - start > 500000:  # Skip if empty or too large (>500KB)
                        continue
                        
                    # Add the section with its heading
                    section_text = text[start:end].strip()
                    if section_text and len(section_text) > 50:  # Only add substantial sections
                        sections.append(section_text)
                
                if len(sections) >= 2:
                    # Merge tiny sections if needed (but with size limit)
                    if len(sections) > 20:
                        logger.info(f"Found {len(sections)} sections, merging some")
                        merged_sections = []
                        current = sections[0]
                        
                        for i in range(1, len(sections)):
                            # Only merge if the result is still manageable
                            if len(current) + len(sections[i]) <= min(self.chunk_size * 2, 100000):
                                current += "\n\n" + sections[i]
                            else:
                                merged_sections.append(current)
                                current = sections[i]
                        
                        if current:
                            merged_sections.append(current)
                            
                        return merged_sections
                    return sections
        except Exception as e:
            logger.error(f"Error in section splitting: {e}")
            # Continue to fallback method
            
        # Fallback: Simple paragraph-based chunking
        logger.info("Using fallback paragraph-based sectioning")
        try:
            # Split by blank lines
            paragraphs = re.split(r'\n\s*\n', text)
            
            # Aim for approximately 5-10 sections
            target_section_count = min(10, max(5, len(paragraphs) // 20))
            
            # If very few paragraphs, use them directly
            if len(paragraphs) <= target_section_count:
                return [p for p in paragraphs if p.strip()]
                
            # Combine paragraphs into reasonable sections
            sections = []
            current_section = []
            paragraphs_per_section = max(1, len(paragraphs) // target_section_count)
            
            for i, para in enumerate(paragraphs):
                if not para.strip():
                    continue
                    
                current_section.append(para)
                
                if (i + 1) % paragraphs_per_section == 0 or i == len(paragraphs) - 1:
                    combined = "\n\n".join(current_section)
                    if combined.strip():
                        sections.append(combined)
                    current_section = []
            
            return sections
            
        except Exception as e:
            logger.error(f"Error in fallback section splitting: {e}")
            
            # Ultimate fallback: just split into roughly equal chunks
            text_len = len(text)
            chunk_count = max(2, min(10, text_len // 10000))  # 1-10 chunks based on size
            chunk_size = text_len // chunk_count
            
            sections = []
            for i in range(chunk_count):
                start = i * chunk_size
                end = start + chunk_size if i < chunk_count - 1 else text_len
                chunk = text[start:end].strip()
                if chunk:
                    sections.append(chunk)
            
            return sections
    
    def _extract_section_title(self, section_text: str) -> str:
        """Extract the title of a section from its text"""
        # Try to find the first line that looks like a heading
        lines = section_text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Check if line is a potential heading (not too long, has content)
            if line and len(line) < 100 and not line.endswith('.'):
                # Look for heading patterns
                if (re.match(r'^#+\s+', line) or  # Markdown style
                    re.match(r'^\d+\.\d*\s+', line) or  # Numbered heading
                    re.match(r'^[IVXLC]+\.\s+', line) or  # Roman numerals
                    re.match(r'^[A-Z][A-Z\s]+$', line) or  # ALL CAPS
                    re.match(r'^(?:CHAPTER|SECTION|ARTICLE)\s+\d+', line.upper())):  # Chapter/Section
                    
                    # Clean up markdown symbols and section numbers
                    cleaned = re.sub(r'^#+\s+', '', line)
                    cleaned = re.sub(r'^\d+\.\d*\s+', '', cleaned)
                    cleaned = re.sub(r'^[IVXLC]+\.\s+', '', cleaned)
                    cleaned = re.sub(r'^(?:CHAPTER|SECTION|ARTICLE)\s+\d+[:\s]+', '', cleaned, flags=re.IGNORECASE)
                    
                    return cleaned
                    
                # First line is likely the title if it's short enough
                if len(line) < 80:
                    return line
        
        # Default to first 50 characters if no heading found
        return section_text[:50].strip() + '...'
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks by sentences, respecting chunk size"""
        sentences = self._safe_sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If sentence itself exceeds chunk size, split it further
                if len(sentence) > self.chunk_size:
                    sentence_chunks = self._split_long_sentence(sentence)
                    chunks.extend(sentence_chunks[:-1])  # Add all but the last chunk
                    current_chunk = sentence_chunks[-1]  # Start with the last chunk
                else:
                    current_chunk = sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split an extremely long sentence into smaller parts"""
        # Try to split by common separators
        for separator in [',', ';', ':', ')', '(', '-', '—']:
            if separator in sentence:
                parts = sentence.split(separator)
                if all(len(part) < self.chunk_size for part in parts):
                    result = []
                    current = ""
                    
                    for i, part in enumerate(parts):
                        sep = separator if i > 0 else ""
                        if len(current) + len(sep) + len(part) <= self.chunk_size:
                            current += sep + part
                        else:
                            result.append(current)
                            current = part
                    
                    if current:
                        result.append(current)
                        
                    return result
        
        # If no good separator, split by words
        words = sentence.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = word
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def apply_paragraph_chunking(self, doc: Document = None) -> List[Document]:
        """
        Apply paragraph-based chunking to documents
        
        Args:
            doc: Optional single document to chunk (if not provided, processes all documents)
            
        Returns:
            List of Document objects after chunking
        """
        if doc is not None:
            # Process a single document
            documents_to_process = [doc]
        else:
            # Process all documents
            documents_to_process = self.documents
            
        chunked_docs = []
        
        for doc in documents_to_process:
            paragraphs = re.split(r'\n\s*\n', doc.text)
            
            current_chunk = ""
            doc_chunks = []
            
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if not para:
                    continue
                    
                # If adding this paragraph would exceed the chunk size
                if len(current_chunk) + len(para) + 2 > self.chunk_size:
                    # Save current chunk if not empty
                    if current_chunk:
                        doc_chunks.append(current_chunk)
                    
                    # If paragraph itself exceeds chunk size, split it
                    if len(para) > self.chunk_size:
                        # Split paragraph by sentences
                        sentences = self._safe_sent_tokenize(para)
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                                if current_chunk:
                                    current_chunk += " " + sentence
                                else:
                                    current_chunk = sentence
                            else:
                                if current_chunk:
                                    doc_chunks.append(current_chunk)
                                
                                # If sentence is itself too long, split it
                                if len(sentence) > self.chunk_size:
                                    chunks = self._split_long_sentence(sentence)
                                    doc_chunks.extend(chunks[:-1])
                                    current_chunk = chunks[-1]
                                else:
                                    current_chunk = sentence
                    else:
                        current_chunk = para
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            # Add the last chunk if not empty
            if current_chunk:
                doc_chunks.append(current_chunk)
                
            # Create documents from chunks
            for i, chunk in enumerate(doc_chunks):
                chunked_docs.append(Document(
                    text=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "paragraph",
                        "chunk_index": i
                    }
                ))
                
            logger.info(f"Created {len(doc_chunks)} paragraph-based chunks from document")
            
        return chunked_docs
    
    def apply_fixed_chunking(self, doc: Document = None) -> List[Document]:
        """
        Apply fixed-size chunking to documents
        
        Args:
            doc: Optional single document to chunk (if not provided, processes all documents)
            
        Returns:
            List of Document objects after chunking
        """
        if doc is not None:
            # Process a single document
            documents_to_process = [doc]
        else:
            # Process all documents
            documents_to_process = self.documents
            
        chunked_docs = []
        
        for doc in documents_to_process:
            fixed_chunks = self._chunk_by_fixed_size(doc.text)
            
            doc_chunks = [
                Document(
                    text=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_type": "fixed", 
                        "chunk_index": i
                    }
                ) for i, chunk in enumerate(fixed_chunks)
            ]
            
            chunked_docs.extend(doc_chunks)
            logger.info(f"Created {len(doc_chunks)} fixed-size chunks from document")
            
        return chunked_docs
    
    def _chunk_by_fixed_size(self, text: str) -> List[str]:
        """Split text into chunks of fixed size with overlap"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            # Determine end position for this chunk
            end = start + self.chunk_size
            
            # If we're beyond the text, just take the remainder
            if end >= text_len:
                chunks.append(text[start:])
                break
                
            # Try to find a good breaking point near the end
            # First try a paragraph break
            good_break = text.rfind('\n\n', start, end)
            
            if good_break == -1 or good_break < start + (self.chunk_size // 2):
                # If no paragraph break or it would make chunk too small,
                # try to find a sentence break
                good_break = text.rfind('. ', start, end)
                
                if good_break == -1 or good_break < start + (self.chunk_size // 2):
                    # If no sentence break or it would make chunk too small,
                    # use space as last resort
                    good_break = text.rfind(' ', start, end)
                    
                    if good_break == -1 or good_break < start + (self.chunk_size // 2):
                        # If all else fails, break at the exact position
                        good_break = end
            
            # Extract the chunk and add to result
            chunks.append(text[start:good_break].strip())
            
            # Update start position for next chunk, accounting for overlap
            start = good_break - self.chunk_overlap if self.chunk_overlap > 0 else good_break
            
            # Ensure we're making forward progress
            if start <= 0 or start <= good_break - self.chunk_size:
                start = good_break + 1
                
        return chunks
    
    def create_vector_store(self, chunked_docs: List[Document]) -> Tuple[VectorStoreIndex, ChromaVectorStore]:
        """Create a vector store index from chunked documents with fixed 512d embeddings"""
        # Clean existing database if it exists and clear_chroma is True
        if self.clear_chroma and os.path.exists(self.chroma_dir):
            logger.info(f"Removing existing Chroma database at {self.chroma_dir}")
            shutil.rmtree(self.chroma_dir)
        
        # Configure settings with our embedding model
        Settings.embed_model = self.embed_model
        
        # Clean documents to avoid embedding errors
        cleaned_docs = []
        for doc in chunked_docs:
            try:
                # Clean text to avoid embedding issues
                cleaned_text = self.clean_text(doc.text)
                if cleaned_text.strip():
                    cleaned_docs.append(Document(
                        text=cleaned_text,
                        metadata=doc.metadata
                    ))
            except Exception as e:
                logger.error(f"Error cleaning document: {str(e)}")
        
        logger.info(f"Proceeding with {len(cleaned_docs)} cleaned documents for vector store")
        
        try:
            # Create Chroma collection
            chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
            collection_name = f"financial_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            chroma_collection = chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_dim": 512,  # Fixed at 512
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }
            )
            
            # Create vector store
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index with batched processing and error handling
            try:
                # Process in smaller batches to avoid memory issues
                batch_size = 50  # Process 50 documents at a time
                
                if len(cleaned_docs) <= batch_size:
                    # Small enough to process in one go
                    index = VectorStoreIndex.from_documents(
                        cleaned_docs,
                        storage_context=storage_context,
                        show_progress=True
                    )
                    logger.info(f"Created vector index with {len(cleaned_docs)} chunks")
                else:
                    # Process in batches
                    logger.info(f"Processing {len(cleaned_docs)} documents in batches of {batch_size}")
                    
                    # First batch creates the index
                    first_batch = cleaned_docs[:batch_size]
                    index = VectorStoreIndex.from_documents(
                        first_batch,
                        storage_context=storage_context,
                        show_progress=True
                    )
                    logger.info(f"Created initial index with {len(first_batch)} chunks")
                    
                    # Process remaining batches
                    remaining_docs = cleaned_docs[batch_size:]
                    for i in range(0, len(remaining_docs), batch_size):
                        batch = remaining_docs[i:i+batch_size]
                        logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} chunks")
                        
                        # Insert additional documents
                        for doc in batch:
                            index.insert(doc)
                
                return index, vector_store
                
            except Exception as e:
                logger.error(f"Error creating vector index: {str(e)}")
                logger.info("Trying with smaller batch and simpler processing...")
                
                # Retry with simpler approach and smaller batch
                if len(cleaned_docs) > 20:
                    subset_docs = cleaned_docs[:20]
                    index = VectorStoreIndex.from_documents(
                        subset_docs,
                        storage_context=storage_context,
                        show_progress=True
                    )
                    logger.warning(f"Created partial vector index with only {len(subset_docs)} chunks")
                    return index, vector_store
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"Error in vector store creation: {str(e)}")
            raise
    
    def process(self) -> tuple:
        """
        Process documents end-to-end: load, chunk, and create vector store
        
        Returns:
            tuple: (index, chunked_docs, vector_store)
        """
        try:
            # Load documents
            self.load_documents()
            
            # Process documents with chosen chunking strategy
            chunked_docs = self.process_documents()
            
            # Create vector store
            index, vector_store = self.create_vector_store(chunked_docs)
            
            return index, chunked_docs, vector_store
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            
            # Try to recover some chunked documents even if vector store creation fails
            if 'chunked_docs' in locals() and chunked_docs:
                logger.error(f"Returning {len(chunked_docs)} chunks without vector index")
                return None, chunked_docs, None
            
            # If chunking also failed, create simple chunks
            if self.documents:
                logger.error("Creating simple chunks as fallback")
                simple_chunks = []
                for doc in self.documents:
                    # Split by paragraphs
                    paragraphs = re.split(r'\n\s*\n', doc.text)
                    for i, para in enumerate(paragraphs):
                        if para.strip():
                            simple_chunks.append(Document(
                                text=para.strip(),
                                metadata={**doc.metadata, "chunk_type": "simple", "chunk_index": i}
                            ))
                return None, simple_chunks, None
            
            # If all else fails
            raise