from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_recursive(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_text(text.strip())
    return [chunk for chunk in chunks if chunk.strip()]
