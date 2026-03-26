from unittest.mock import MagicMock, patch

import upscale_video


def test_spinner_rich_available():
    # Force rich to be available
    with patch("upscale_video._RICH_AVAILABLE", True):
        # Mock the console status
        mock_console = MagicMock()
        mock_status = MagicMock()
        mock_console.status.return_value = mock_status
        with patch("upscale_video._console", mock_console):
            with upscale_video.spinner("Test Spinner"):
                pass
            mock_console.status.assert_called_once_with("Test Spinner", spinner="dots")

def test_spinner_rich_unavailable():
    with patch("upscale_video._RICH_AVAILABLE", False):
        # Should just be a nullcontext, no errors
        with upscale_video.spinner("Test Spinner"):
            pass

def test_frame_progress_rich_available():
    with patch("upscale_video._RICH_AVAILABLE", True):
        mock_progress = MagicMock()
        mock_task = MagicMock()
        mock_progress.add_task.return_value = mock_task
        
        # Need to mock Progress class initialization and context manager
        mock_progress.__enter__.return_value = mock_progress
        
        with patch("upscale_video.Progress", return_value=mock_progress):
            with upscale_video.frame_progress(10, "Testing") as advance:
                advance()
        
            mock_progress.add_task.assert_called_once_with("Testing", total=10)
            mock_progress.advance.assert_any_call(mock_task)

def test_frame_progress_rich_unavailable():
    with patch("upscale_video._RICH_AVAILABLE", False):
        with upscale_video.frame_progress(10, "Testing") as advance:
            advance()
            # Should not raise any errors, and return a callable noop
            assert callable(advance)
