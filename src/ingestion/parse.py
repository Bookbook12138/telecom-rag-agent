from pathlib import Path
from pypdf import PdfReader

def load_pdf_folder(folder_path: Path):
    """读取文件夹内所有pdf，返回 (文件名, 全文文本) 的列表"""
    results = []
    if not folder_path.exists():
        print(f"警告：文件夹 {folder_path} 不存在！")
        return results

    for pdf_file in folder_path.glob("*.pdf"):
        reader = PdfReader(pdf_file)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        results.append((pdf_file.name, full_text))
    return results
