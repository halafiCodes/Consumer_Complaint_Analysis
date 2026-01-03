def clean_text(text: str) -> str:
    """
    Cleans a complaint narrative by lowercasing, removing special characters, 
    and normalizing whitespace.

    Args:
        text (str): Raw complaint narrative.

    Returns:
        str: Cleaned text suitable for embedding.
    """
