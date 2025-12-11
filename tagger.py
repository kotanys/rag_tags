import os.path
import sys
import time
import concurrent.futures
from io import TextIOWrapper
from typing import Generator

import warnings
# Suppress the specific Pydantic compatibility warning
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible with Python 3.14")

import openai

def paragraph_chunker(text_stream: TextIOWrapper, paragraphs_per_chunk=30) -> Generator[list[str], None, None]:
    """
    Generator that takes a stream of text and yields chunks of approximately 
    paragraphs_per_chunk paragraphs as lists of strings.
    
    Args:
        text_stream: An iterable yielding lines of text or a single string
        paragraphs_per_chunk: Target number of paragraphs per chunk
    
    Yields:
        list[str]: Lists of paragraph strings, each list containing approximately
                  paragraphs_per_chunk paragraphs
    """
    paragraphs = []
    current_paragraph = []
    
    for line in text_stream:
        stripped_line = line.strip()
        
        if not stripped_line:
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        else:
            current_paragraph.append(stripped_line)
    
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    # Now yield chunks of approximately paragraphs_per_chunk paragraphs
    chunk = []
    chunk_size = 0
    
    for paragraph in paragraphs:
        if not paragraph.strip():  # Skip empty paragraphs
            continue
            
        chunk.append(paragraph)
        chunk_size += 1
        
        if chunk_size >= paragraphs_per_chunk:
            yield chunk
            chunk = []
            chunk_size = 0
    
    # Yield any remaining paragraphs
    if chunk:
        yield chunk

def process_chunk(client, prompt_id, chunk, chunk_index):
    """
    Process a single chunk and return result with index.
    """
    max_retries = 5
    while max_retries > 0:
        max_retries -= 1
        try:
            response = client.responses.create(
                prompt={"id": prompt_id},
                input="\n\n".join(chunk),
            )
            output: str = response.output_text
            
            if len(output.strip()) < 10:
                raise Exception()

            if output.endswith('"}]'):
                pass
            if output.endswith('"}'):
                output += ']'
            elif output.endswith('"'):
                output += '}]'
            else:
                output += '"}]'

            return {"index": chunk_index, "text": output, "error": None}
        except Exception:
            print(f"retrying chunk {chunk_index + 1}")
            time.sleep(3)

    print(f"chunk {chunk_index + 1} exhausted it's retries")
    return {"index": chunk_index, "text": None, "error": "failed after 5 retries"}

def process_concurrently_threads(client, prompt_id, chunks, max_workers=3, request_delay=0.5):
    """
    Process chunks concurrently using threads.
    
    Args:
        client: Yandex Cloud client
        prompt_id: Prompt template ID
        chunks: List of chunk lists
        max_workers: Maximum concurrent requests (3-5 is safe)
        request_delay: Delay between starting requests (seconds)
    """
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {}
        for i, chunk in enumerate(chunks):
            # Stagger request starts to avoid rate limiting
            time.sleep(request_delay / max_workers)
            
            future = executor.submit(
                process_chunk, 
                client, prompt_id, chunk, i
            )
            future_to_chunk[future] = i
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✓ Chunk {chunk_index + 1}/{len(chunks)} completed")
            except Exception as e:
                print(f"✗ Chunk {chunk_index + 1} failed: {e}")
                results.append({"index": chunk_index, "text": None, "error": str(e)})
    
    # Sort by original chunk order
    results.sort(key=lambda x: x["index"])
    return results


# Usage example
def main():
    client = openai.OpenAI(
        api_key=os.environ["YANDEX_CLOUD_API_KEY"],
        base_url="https://rest-assistant.api.cloud.yandex.net/v1",
        project=os.environ["YANDEX_CLOUD_FOLDER"]
    )
    
    max_workers=10
    FILE = sys.argv[1]
    if not os.path.exists(FILE):
        print(f"'{FILE}' does not exist")
        return

    print(f"Processing '{FILE}'")

    with open(FILE, "rt", encoding='utf-8') as f:
        # Convert generator to list for concurrent processing
        chunks = list(paragraph_chunker(f))
    
    print(f"Processing {len(chunks)} chunks with {max_workers} concurrent workers...")
    
    results = process_concurrently_threads(
        client=client,
        prompt_id=os.environ["YANDEX_CLOUD_AGENT_ID"],
        chunks=chunks,
        max_workers=max_workers,
    )
      
    to_output: list[str] = []
    # Process results
    for result in results:
        if result["error"]:
            print(f"Chunk {result['index'] + 1} error: {result['error']}")
        else:
            to_output.append(result["text"])
    with open(FILE + ".output", "wt") as output_file:
        output_file.write('[\n')
        output_file.write(',\n'.join(to_output))
        output_file.write(']')

if __name__ == "__main__":
    main()
