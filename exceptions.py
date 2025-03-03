class VideoNotFoundError(Exception):
    """Raised when the requested video is not found"""
    def __init__(self, video_id: int):
        self.video_id = video_id
        self.message = f"Video with ID {video_id} not found"
        super().__init__(self.message)

class DatabaseConnectionError(Exception):
    """Raised when there is a failure connecting to the database"""
    def __init__(self, message="Database connection failed"):
        self.message = message
        super().__init__(self.message)
