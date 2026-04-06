# PDF to Markdown Converter

A simple Python tool to convert PDF files to Markdown format.

## Installation

Install the required dependency:

```bash
pip install pymupdf
```

Or install all dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Conversion

Convert a PDF file to Markdown (creates output file with same name):

```bash
python pdf_to_md.py document.pdf
```

This will create `document.md` in the same directory.

### Specify Output File

Convert to a specific output file:

```bash
python pdf_to_md.py document.pdf -o output.md
```

### Overwrite Existing Files

If the output file already exists, use the `--overwrite` flag:

```bash
python pdf_to_md.py document.pdf --overwrite
```

### Example with your PDF

Convert the PhD thesis in your project:

```bash
python pdf_to_md.py 2023TaoPhD.pdf
```

This will create `2023TaoPhD.md` with the extracted text.

## Features

- Extracts text from all pages
- Preserves PDF metadata (title, author, subject)
- Organizes content by page numbers
- Cleans up excessive whitespace
- Simple command-line interface

## Using as a Python Module

You can also import and use the converter in your Python code:

```python
from pdf_to_md import convert_pdf_to_markdown

# Convert a PDF
output_file = convert_pdf_to_markdown('document.pdf')
print(f"Created: {output_file}")

# With custom output path
convert_pdf_to_markdown('document.pdf', output_path='custom.md', overwrite=True)
```

## Limitations

- Text-only extraction (images are not included)
- Complex layouts may not preserve exact formatting
- Tables are extracted as plain text
- Best results with text-based PDFs (not scanned images)

## Troubleshooting

If you get an import error for `fitz`, make sure PyMuPDF is installed:

```bash
pip install pymupdf
```

Note: The package is `pymupdf` but imports as `fitz`.
