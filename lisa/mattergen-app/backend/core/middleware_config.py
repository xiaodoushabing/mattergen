import time
from fastapi import Request
from loguru import logger
from core.logging_config import get_logger
import json
from starlette.middleware.base import BaseHTTPMiddleware

logger = get_logger(core="middleware")

def format_request_log(request, response_status=None, error=None):
    try:
        log_dict = {
            "url": str(request.url),
            "method": request.method,
            "client_ip": request.client.host
        }

        headers = {k: v for k, v in request.headers.items()}
        if headers:
            log_dict["headers"] = headers
        
        if response_status is not None:
            log_dict["status_code"] = response_status
            
        if error:
            log_dict["error"] = str(error)
            
        return json.dumps(log_dict, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "error": f"Error formatting log: {str(e)}",
            "url": str(getattr(request, 'url', 'Unknown')),
            "method": str(getattr(request, 'method', 'Unknown'))
        })

class LogRequestMiddleware(BaseHTTPMiddleware): 
    async def dispatch(self, request: Request, call_next):
        """
        Middleware to log incoming requests and their processing time.
        """
        start_time = time.time()
        request_id = str(time.time())
        
        try:
            logger.debug(f"middleware | Request : {request}")
            request_log = format_request_log(request)
            logger.info(f"middleware | Request started | ID: {request_id} | {request_log}")
            
            response = await call_next(request)
            
            process_time = (time.time() - start_time) * 1000
            response_log = format_request_log(request, response_status=response.status_code)
            
            logger.info(f"Request completed | ID: {request_id} | Time: {process_time:.2f}ms | {response_log}")
            return response
            
        except Exception as e:
            process_time = (time.time() - start_time) * 1000
            logger.error(
                f"Request failed | ID: {request_id} | Time: {process_time:.2f}ms | Error: {str(e)}"
            )
            raise