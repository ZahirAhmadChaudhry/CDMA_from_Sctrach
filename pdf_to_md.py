#!/usr/bin/env python3
"""
PDF to Markdown Converter

Converts PDF files to Markdown format, extracting text and preserving structure.
"""

import argparse
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF is not installed.")
    print("Install it with: pip install pymupdf")
    sys.exit(1)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF and format as Markdown."""
    doc = fitz.open(pdf_path)
    markdown_content = []
    
    # Add document title
    pdf_name = Path(pdf_path).stem
    markdown_content.append(f"# {pdf_name}\n")
    
    # Extract metadata if available
    metadata = doc.metadata
    if metadata:
        if metadata.get('title') and metadata['title'] != pdf_name:
            markdown_content.append(f"**Title:** {metadata['title']}\n")
        if metadata.get('author'):
            markdown_content.append(f"**Author:** {metadata['author']}\n")
        if metadata.get('subject'):
            markdown_content.append(f"**Subject:** {metadata['subject']}\n")
        markdown_content.append("\n---\n")
    
    # Process each page
    for page_num, page in enumerate(doc, start=1):
        markdown_content.append(f"\n## Page {page_num}\n")
        
        # Extract text
        text = page.get_text()
        
        # Clean up text: remove excessive newlines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        # Join lines with proper spacing
        page_text = '\n\n'.join(cleaned_lines)
        markdown_content.append(page_text)
        markdown_content.append("\n")
    
    doc.close()
    return '\n'.join(markdown_content)


def convert_pdf_to_markdown(pdf_path: str, output_path: str = None, overwrite: bool = False) -> str:
    """
    Convert a PDF file to Markdown format.
    
    Args:
        pdf_path: Path to input PDF file
        output_path: Path to output Markdown file (optional)
        overwrite: Whether to overwrite existing output file
    
    Returns:
        Path to the created Markdown file
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_file.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    # Determine output path
    if output_path is None:
        output_path = pdf_file.with_suffix('.md')
    else:
        output_path = Path(output_path)
    
    # Check if output file exists
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            "Use --overwrite flag to replace it."
        )
    
    # Extract text and convert to Markdown
    print(f"Converting {pdf_path}...")
    markdown_text = extract_text_from_pdf(str(pdf_file))
    
    # Write to file
    output_path.write_text(markdown_text, encoding='utf-8')
    print(f"✓ Successfully converted to: {output_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Convert PDF files to Markdown format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single PDF
  python pdf_to_md.py document.pdf
  
  # Specify output file
  python pdf_to_md.py document.pdf -o output.md
  
  # Overwrite existing file
  python pdf_to_md.py document.pdf --overwrite
        """
    )
    
    parser.add_argument('pdf_file', help='Path to PDF file to convert')
    parser.add_argument(
        '-o', '--output',
        help='Output Markdown file path (default: same name as PDF with .md extension)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output file if it exists'
    )
    
    args = parser.parse_args()
    
    try:
        convert_pdf_to_markdown(args.pdf_file, args.output, args.overwrite)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
