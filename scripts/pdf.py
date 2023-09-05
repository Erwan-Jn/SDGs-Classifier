from PyPDF2 import PdfReader

def pdf(doc):
    with open(doc, 'rb') as f:
        pdf = PdfReader(f)
        number_of_pages = len(pdf.pages)
        result = []
        for x in range(number_of_pages):
            result.append(pdf.pages[x].extract_text())
        result = ". ".join(result)
        result = result.lower()
    return result
