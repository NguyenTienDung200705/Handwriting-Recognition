let currentFile = null;

document.getElementById("fileInput").addEventListener("change", function(e) {
    const file = e.target.files[0];
    if (!file) return;
    currentFile = file;

    const reader = new FileReader();
    reader.onload = ev => document.getElementById("originalImage").src = ev.target.result;
    reader.readAsDataURL(file);

    // Reset
    document.getElementById("detectionImage").src = "";
    for (let i = 1; i <= 4; i++) {
        const box = document.getElementById(`result${i}`);
        if (box) {
            box.innerHTML = `<pre class="placeholder-text">Kết quả OCR ${i} sẽ hiển thị tại đây...</pre>`;
        }
    }
});

async function runDetect() {
    if (!currentFile) return alert("❌ Vui lòng chọn ảnh trước!");

    const formData = new FormData();
    formData.append("image", currentFile);

    try {
        const res = await fetch("/detect", { method: "POST", body: formData });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        document.getElementById("detectionImage").src =
            "data:image/jpeg;base64," + data.image;

    } catch (err) {
        alert("❌ Lỗi Detect: " + err.message);
    }
}

async function runOCR(arch, resultId) {
    if (!currentFile) return alert("❌ Vui lòng chọn ảnh trước!");

    // ✅ Danh sách model hợp lệ (đúng với backend)
    const validModels = ["crnn_vgg", "crnn", "crnn_efficientNet", "crnn_resnet"];

    if (!validModels.includes(arch)) {
        alert(`❌ Model không hợp lệ: ${arch}`);
        console.error("Model sai:", arch);
        return;
    }

    console.log(`[DEBUG] Run OCR với model: "${arch}"`);

    const formData = new FormData();
    formData.append("image", currentFile);

    // ✅ GIỮ NGUYÊN theo backend của bạn
    formData.append("architecture", arch);

    try {
        const res = await fetch("/ocr", { 
            method: "POST", 
            body: formData 
        });

        if (!res.ok) {
            const errorData = await res.json().catch(() => ({}));
            const errorMsg = errorData.error || `Lỗi server (${res.status})`;
            console.error("OCR Error:", errorMsg);
            throw new Error(errorMsg);
        }

        const data = await res.json();

        // ✅ Hiển thị ảnh detect + OCR
        if (data.image) {
            document.getElementById("detectionImage").src = 
                "data:image/jpeg;base64," + data.image;
        }

        const resultBox = document.getElementById(resultId);

        if (resultBox && data.lines && data.lines.length > 0) {
            const text = data.lines
                .map(line => line.trim())
                .filter(line => line.length > 0)
                .join("\n\n");

            resultBox.innerHTML = `
                <pre style="
                    font-family: 'Courier New', monospace;
                    font-size: 1rem;
                    line-height: 1;
                    color: #e0f2e9;
                    white-space: pre-wrap;
                    margin: 0;
                    padding: 15px;
                    background: rgba(0, 0, 0, 0.35);
                    border-radius: 8px;
                ">${text}</pre>
            `;
        } else if (resultBox) {
            resultBox.innerHTML = `
                <pre style="color: #ff9999; padding: 15px;">
Không nhận diện được văn bản nào.
                </pre>
            `;
        }

    } catch (err) {
        console.error("runOCR Error:", err);
        alert("❌ Lỗi OCR: " + err.message);
    }
}