'''Parse Google Docs JSON content'''

def convert_to_text(content: dict) -> str:
    '''Convert Google Doc JSON to plain text'''
    doc_text = ''
    for item in content:
        if 'paragraph' in item:
            elements = item['paragraph']['elements']
            for element in elements:
                if 'textRun' in element:
                    doc_text += element['textRun']['content']

    return doc_text

def _convert_paragraph(elements):
    '''Convert a list of elements in a paragraph to markdown'''
    paragraph_md = ''
    for element in elements:
        if 'textRun' in element:
            text_run = element['textRun']
            element_md = text_run.get('content', '')

            # Handle bold formatting
            if text_run.get('textStyle', {}).get('bold'):
                element_md = f'**{element_md}**'

            # Handle italic formatting
            if text_run.get('textStyle', {}).get('italic'):
                element_md = f'*{element_md}*'

            paragraph_md += element_md
    return paragraph_md

def convert_to_markdown(content: dict):
    '''Convert Google Doc JSON content to markdown'''
    markdown_paragraphs = [] # Output paragraphs
    prev_list_item = ''      # Remember if the previous paragraph was a list item (to avoid adding a new line between list items)
    for item in content:
        if 'paragraph' in item:
            paragraph_md = '' # Markdown for the current paragraph
            paragraph = item['paragraph']

            # Handle headings
            if 'paragraphStyle' in paragraph:
                style = paragraph['paragraphStyle']
                heading_type = style.get('namedStyleType', '')
                if 'HEADING' in heading_type:
                    # Add markdown syntax for headings
                    level = int(heading_type[-1])  # Extract heading level
                    paragraph_md += '#' * level + ' '

            elements = paragraph.get('elements', [])

            # Detect if the paragraph is part of a list
            if 'bullet' in paragraph:
                # TODO: would be nice to detect ordered lists,
                # but there doesn't seem to be any indication of order in the json

                # Add the bullet or numbered list item
                paragraph_md += f'* {_convert_paragraph(elements)}'
                if prev_list_item:
                    # If the previous paragraph was also a list item,
                    # append to it instaed of starting a new paragraph
                    markdown_paragraphs[-1] = markdown_paragraphs[-1] + paragraph_md
                else:
                    # Otherwise this is a new paragraph
                    markdown_paragraphs.append(paragraph_md)
                prev_list_item = paragraph_md
            else:
                # Handle regular paragraphs
                paragraph_md += _convert_paragraph(elements)
                if paragraph_md.strip():
                    # Only add non-empty paragraphs
                    markdown_paragraphs.append(paragraph_md)
                    prev_list_item = ''

    return '\n'.join(markdown_paragraphs)

