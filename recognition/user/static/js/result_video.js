document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('videoPlayer');
    const downloadBtn = document.getElementById('downloadBtn');

    let videoError = false;
    let playAttempted = false;

    video.addEventListener('loadeddata', function() {
        console.log('Video data loaded successfully');
    });

    video.addEventListener('playing', function() {
        console.log('Video playback started');
        playAttempted = true;
    });

    video.addEventListener('error', function(e) {
        console.error('Video playback error:', e);
        videoError = true;
        highlightDownloadButton();
    });

    setTimeout(function() {
        if (!playAttempted) {
            try {
                const playPromise = video.play();
                if (playPromise !== undefined) {
                    playPromise.then(() => {
                        setTimeout(() => {
                            video.pause();
                        }, 500);
                    }).catch(error => {
                        console.error("Autoplay test failed:", error);
                        highlightDownloadButton();
                    });
                }
            } catch (e) {
                console.error("Error during autoplay test:", e);
            }
        }
    }, 1000);

    setTimeout(function() {
        if (video.readyState === 0 && !videoError && !playAttempted) {
            highlightDownloadButton();
        }
    }, 3000);

    function highlightDownloadButton() {
        downloadBtn.classList.add('highlighted-button');
        downloadBtn.innerHTML = '<i class="fas fa-download"></i> <strong>Download video</strong>';
    }
});