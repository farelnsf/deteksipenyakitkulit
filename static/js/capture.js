document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('captureBtn');
    const imageInput = document.getElementById('imageInput');
    const previewImage = document.getElementById('previewImage');
    const detectBtn = document.getElementById('detectBtn');
    const form = document.getElementById('form');
    const cameraSelect = document.getElementById('cameraSelect');
    const ctx = canvas.getContext('2d');
    let stream = null;

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
    }

    async function startCamera(deviceId) {
        stopCamera();
        
        try {
            const constraints = {
                video: {
                    deviceId: deviceId ? { exact: deviceId } : undefined,
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                }
            };

            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            video.play();
        } catch (err) {
            console.error("Error accessing camera:", err);
            alert('Gagal mengakses kamera. Pastikan Anda memberikan izin akses kamera.');
        }
    }

    async function setupCameras() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            
            cameraSelect.innerHTML = '';
            videoDevices.forEach((device, i) => {
                cameraSelect.add(new Option(device.label || `Kamera ${i+1}`, device.deviceId));
            });
            
            if (videoDevices.length > 0) {
                await startCamera(videoDevices[0].deviceId);
            }
            
            cameraSelect.addEventListener('change', async () => {
                await startCamera(cameraSelect.value);
            });
        } catch (err) {
            console.error("Error enumerating devices:", err);
            cameraSelect.innerHTML = '<option value="">Tidak dapat mengakses kamera</option>';
        }
    }

    captureBtn.onclick = function() {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
        imageInput.value = dataUrl;
        previewImage.src = dataUrl;
        previewImage.style.display = 'block';

        video.style.display = 'none';
        captureBtn.style.display = 'none';
        cameraSelect.style.display = 'none';
        detectBtn.style.display = 'inline-block';

        stopCamera();
    };

    form.addEventListener('submit', function() {
        detectBtn.disabled = true;
        detectBtn.textContent = 'Memproses...';
    });

    window.addEventListener('beforeunload', stopCamera);
    setupCameras();
});