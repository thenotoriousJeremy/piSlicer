import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from math import ceil, sqrt
import os

def merge_pdf_to_one_page(input_pdf, output_pdf):
    # Open the PDF
    doc = fitz.open(input_pdf)
    num_pages = len(doc)

    # Calculate grid size
    cols = ceil(sqrt(num_pages))  # Approximate square root for grid width
    rows = ceil(num_pages / cols)  # Calculate the required rows based on columns

    # Get the dimensions of the first page
    first_page = doc[0]
    page_width, page_height = first_page.rect.width, first_page.rect.height

    # Calculate the dimensions of the final canvas
    canvas_width = cols * page_width
    canvas_height = rows * page_height

    # Create a new PDF canvas
    c = canvas.Canvas(output_pdf, pagesize=(canvas_width, canvas_height))

    # Place each page on the canvas
    x_offset, y_offset = 0, canvas_height - page_height  # Start at top-left
    temp_images = []

    for i, page in enumerate(doc):
        # Render each page as an image
        pix = page.get_pixmap()
        image_path = f"temp_page_{i}.png"
        pix.save(image_path)
        temp_images.append(image_path)

        # Draw the image on the canvas
        c.drawImage(image_path, x_offset, y_offset, width=page_width, height=page_height)

        # Update offsets
        x_offset += page_width
        if x_offset >= canvas_width:  # Move to next row
            x_offset = 0
            y_offset -= page_height

    # Save the final PDF
    c.save()

    # Clean up temporary PNGs
    for image in temp_images:
        os.remove(image)

    print(f"PDF successfully merged into one page: {output_pdf}")

# Example usage
merge_pdf_to_one_page("multi_page_pattern.pdf", "single_page_pattern.pdf")
