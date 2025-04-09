import io
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from services.retrieval_service import RetrievalService
from deps import retrieve_lattices
from models.download import DownloadRequest
from services.download_service import format_lattice_extxyz

from core.logging_config import get_logger
logger = get_logger(route="download")

router = APIRouter()

@router.post("/lattices/download", response_class=StreamingResponse)
async def download_lattices(request: DownloadRequest,
                            retrieval_service: RetrievalService = Depends(retrieve_lattices)):
    """
    Downloads a combined .extxyz file for the specified lattice ID(s).

    Retrieves lattice data from the database for each ID provided in the
    request body, formats them into EXTXYZ format, concatenates them,
    and streams the resulting file back to the client.

    Args:
        request (DownloadRequest): Request body containing:
            - lattice_ids (List[str]): List of MongoDB ObjectIds (as strings)
                                       of the lattices to include.
            - filename (str): Desired filename for the download (e.g., "my_lattices.extxyz").
        retrieval_service (RetrievalService): Injected dependency for fetching lattice data.

    Returns:
        StreamingResponse: A response that streams the generated .extxyz file
                           content to the client.
    """
    lattice_ids = request.lattice_ids
    desired_filename = request.filename
    num_lattices = len(lattice_ids)

    logger.info(f"Received request to download {num_lattices} lattices with IDs: {lattice_ids} into {desired_filename}") 

    # Use an in-memory text buffer to build the file content
    string_io = io.StringIO()
    lattices_processed_count = 0
    lattices_failed = []

    for (idx, lattice_id) in enumerate(lattice_ids, start=1):
        logger.debug(f"Processing lattice {idx}/{num_lattices} (ID: {lattice_id})")
        try:
            lattice = retrieval_service.get_lattice_by_id(lattice_id)
            if lattice:
                frame = format_lattice_extxyz(lattice_id, lattice)
                if frame:
                    string_io.write(frame)
                    lattices_processed_count += 1
                    logger.debug(f"Lattice {idx} written successfully to the buffer.")
                else:
                    lattices_failed.append(lattice_id)
                    logger.warning(f"Failed to format lattice {lattice_id} - skipping.")
            else:
                logger.warning(f"Lattice not found: {lattice_id} - skipping.")
                lattices_failed.append(lattice_id)

        except Exception as e:
            logger.error(f"ERROR: Failed retrieving/processing lattice {lattice_id}: {e}", exc_info=True)
            continue
    
    if lattices_processed_count == 0:
        logger.error("No lattices could be processed or found for the download.")
        raise HTTPException(
            status_code=404,
            detail=f"Could not find or process any of the requested lattice IDs: {lattice_ids}."
        ) 
    
    logger.info(f"Finished processing. Successfully formatted: {lattices_processed_count}, Failed/Skipped: {len(lattices_failed)}.")
    
   # Get the complete file content from the buffer
    file_content = string_io.getvalue()
    string_io.close()

    # Encode the string content to bytes (e.g., UTF-8) for the response
    content_bytes = file_content.encode('utf-8')

    # Create an in-memory bytes buffer for StreamingResponse
    bytes_io = io.BytesIO(content_bytes)

    # Set headers for the download
    headers = {
        'Content-Disposition': f'attachment; filename="{desired_filename}"'
    }

    # Return the StreamingResponse
    return StreamingResponse(
        content=bytes_io,
        media_type="chemical/x-xyz", # Official MIME type for XYZ
        # Alternatively use: "application/octet-stream" for generic binary download
        headers=headers
    ) 