// 文档加载完成后执行初始化操作
document.addEventListener('DOMContentLoaded', function() {
    // 获取DOM元素引用
    const dropArea = document.getElementById('drop-area');       // 文件拖放区域
    const fileInput = document.getElementById('image-upload');   // 文件选择输入框
    const predictBtn = document.getElementById('predict-btn');   // 预测按钮
    const resultContainer = document.getElementById('result');  // 结果显示区域
    const emptyState = document.querySelector('.empty-state');  // 空状态提示
    const uploadPlaceholder = document.querySelector('.upload-placeholder'); // 上传占位提示

    // ------------- 文件拖放功能实现 -------------
    // 阻止默认的拖放行为，防止页面跳转
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    // 阻止事件的默认行为和冒泡
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // 当有文件拖入或悬停时，高亮显示拖放区域
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    // 当文件离开或放下时，取消高亮显示
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // 添加高亮样式类
    function highlight() {
        dropArea.classList.add('highlight');
    }

    // 移除高亮样式类
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }

    // 处理文件放下事件
    dropArea.addEventListener('drop', handleDrop, false);

    // 处理拖放的文件，将其赋值给文件输入框并触发change事件
    function handleDrop(e) {
        console.log('Drop event triggered');

        // 检查是否有文件被拖放
        if (!e.dataTransfer || !e.dataTransfer.files || e.dataTransfer.files.length === 0) {
            showError('未检测到拖放的文件');
            return;
        }

        const dt = e.dataTransfer;
        const file = dt.files[0]; // 获取第一个拖放的文件

        if (file && isImageFile(file)) {
            // 创建一个DataTransfer对象来模拟文件列表
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);

            // 将文件设置到文件输入框中（修正的实现）
            fileInput.files = dataTransfer.files;

            // 触发change事件，通知文件已选择
            fileInput.dispatchEvent(new Event('change', { bubbles: true }));
        } else if (file) {
            showError('请上传JPG、PNG或GIF格式的图片');
        } else {
            showError('不支持的文件类型或空文件');
        }
    }

    // ------------- 文件选择样式处理 -------------
    // 监听文件选择输入框的变化
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            const file = this.files[0];
            if (isImageFile(file)) {
                updateUploadPlaceholder(file);
            } else {
                showError('请上传JPG、PNG或GIF格式的图片');
                this.value = ''; // 清除无效文件
            }
        }
    });

    // 检查是否为图像文件
    function isImageFile(file) {
        // 检查文件类型是否匹配图像MIME类型
        return file.type.match('image.*');
    }

    // 更新上传区域的显示内容，显示已选择的文件信息和预览
    function updateUploadPlaceholder(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            uploadPlaceholder.innerHTML = `
                <div class="preview-container">
                    <img src="${e.target.result}" alt="图片预览" class="preview-image">
                    <div class="preview-info">
                        <p>已选择: ${file.name}</p>
                        <p class="file-size">${formatFileSize(file.size)}</p>
                    </div>
                </div>
            `;
        };
        reader.onerror = function() {
            showError('无法读取图片文件，请尝试其他图片');
            uploadPlaceholder.innerHTML = `
                <i class="fas fa-cloud-upload-alt fa-4x"></i>
                <p>拖放图片到这里或点击浏览</p>
                <p class="file-types">支持 JPG、PNG、GIF 格式</p>
            `;
        };
        reader.readAsDataURL(file);
    }

    // 格式化文件大小，将字节转换为更易读的单位
    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        else return (bytes / 1048576).toFixed(1) + ' MB';
    }

    // 获取CSS变量的值，用于动态设置样式
    function getCSSVariable(variableName) {
        return getComputedStyle(document.documentElement).getPropertyValue(variableName);
    }

    // 显示错误信息
    function showError(message) {
        resultContainer.innerHTML = `
            <div class="error-message">
                <i class="fas fa-exclamation-circle"></i>
                <p>${message}</p>
            </div>
        `;
    }

    // 显示加载状态
    function showLoading() {
        resultContainer.innerHTML = `
            <div class="loading-container">
                <div class="loading"></div>
                <p class="loading-text">正在分析图片... 请稍候</p>
            </div>
        `;
    }

    // ------------- 预测按钮点击事件处理 -------------
    // 监听预测按钮的点击事件
    predictBtn.addEventListener('click', async function() {
        const file = fileInput.files[0];
        const currentRequestId = Date.now();  // 生成当前请求的唯一ID

        // 检查是否选择了文件
        if (!file) {
            showError('请选择一张图片');
            return;
        }

        // 检查文件大小（限制为10MB）
        const maxSize = 10 * 1024 * 1024;
        if (file.size > maxSize) {
            showError(`图片大小不能超过${maxSize / (1024 * 1024)}MB`);
            return;
        }

        showLoading();
        resultContainer.dataset.currentRequest = currentRequestId;  // 记录当前请求ID

        try {
            // 压缩图片（降低分辨率，保持比例，最大宽度800px）
            const compressedFile = await compressImage(file);

            // 准备表单数据，包含要上传的图片
            const formData = new FormData();
            formData.append('image', compressedFile);

            // 发送图片到后端API进行口罩检测
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData,
                mode: 'cors',
                credentials: 'same-origin'
            });

            // 检查响应状态
            if (!response.ok) {
                throw new Error('网络响应错误 ' + response.statusText);
            }

            // 解析响应JSON数据
            const data = await response.json();

            // 检查是否是当前请求的响应，避免处理过时的响应
            if (resultContainer.dataset.currentRequest != currentRequestId) {
                console.log('忽略过时的响应');
                return;
            }

            // 处理检测结果
            processResult(data, compressedFile, currentRequestId, resultContainer);

        } catch (error) {
            // 确保只处理当前请求的错误
            if (resultContainer.dataset.currentRequest == currentRequestId) {
                console.error('错误:', error);
                // 显示错误信息
                resultContainer.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i>
                        <p>预测过程中发生错误: ${error.message}</p>
                    </div>
                `;
            }
        }
    });

    // ------------- 图片压缩函数 -------------
    function compressImage(file, maxWidth = 800) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = function(e) {
                const img = new Image();
                img.src = e.target.result;
                img.onload = function() {
                    let width = img.width;
                    let height = img.height;

                    // 保持比例压缩图片
                    if (width > maxWidth) {
                        const ratio = maxWidth / width;
                        width = maxWidth;
                        height = height * ratio;
                    }

                    const canvas = document.createElement('canvas');
                    const ctx = canvas.getContext('2d');
                    canvas.width = width;
                    canvas.height = height;

                    ctx.drawImage(img, 0, 0, width, height);

                    canvas.toBlob(blob => {
                        const compressedFile = new File([blob], file.name, {
                            type: file.type,
                            lastModified: Date.now()
                        });
                        resolve(compressedFile);
                    }, file.type, 0.8); // 0.8质量系数
                };
                reader.onerror = reject;
            };
        });
    }

    // ------------- 处理检测结果 -------------
    // 处理API返回的检测结果，在图像上绘制检测框
    function processResult(data, file, requestId, resultContainer) {
        const reader = new FileReader();

        // 文件读取完成后的回调函数
        reader.onload = function(e) {
            // 检查是否是当前请求，避免处理过时的结果
            if (resultContainer.dataset.currentRequest != requestId) {
                console.log('忽略过时的图像处理');
                return;
            }

            // 创建图像对象并设置源
            const img = new Image();
            img.src = e.target.result;
            img.classList.add('result-image');

            // 图像加载完成后的回调函数
            img.onload = function() {
                // 再次检查是否是当前请求
                if (resultContainer.dataset.currentRequest != requestId) {
                    console.log('忽略过时的图像渲染');
                    return;
                }

                // 创建画布，准备在图像上绘制检测框
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');

                // 在画布上绘制原始图像
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

                // 遍历检测结果，绘制每个检测框
                data.forEach(item => {
                    const [x1, y1, x2, y2] = item.bbox;  // 边界框坐标
                    const label = item.label;            // 标签：0表示未戴口罩，1表示已戴口罩
                    const score = item.score;            // 置信度分数
                    const labelName = item.label_name;   // 标签文本

                    // 检查边界框坐标是否有效
                    if (x1 < 0 || y1 < 0 || x2 > img.width || y2 > img.height) {
                        console.warn('无效的边界框坐标:', item.bbox);
                        return;
                    }

                    // 设置边框颜色和标签文本
                    const color = label === 0 ? getCSSVariable('--danger-color') : getCSSVariable('--success-color');

                    // 绘制边界框
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                    // 绘制标签和置信度分数
                    ctx.fillStyle = color;
                    ctx.font = '14px Arial';
                    ctx.fillText(`${labelName} (${score.toFixed(2)})`, x1, y1 - 5);
                });

                // 将绘制好的画布转换为Base64格式的图像
                const base64Image = canvas.toDataURL('image/jpeg');

                // 显示处理后的图像
                if (resultContainer.dataset.currentRequest == requestId) {
                    img.src = base64Image;
                    resultContainer.innerHTML = '';
                    resultContainer.appendChild(img);

                    // 添加检测统计信息
                    if (data.length > 0) {
                        const stats = document.createElement('div');
                        stats.className = 'detection-stats';

                        // 统计已戴口罩和未戴口罩的人数
                        const maskCount = data.filter(item => item.label === 1).length;
                        const noMaskCount = data.length - maskCount;

                        // 显示统计信息
                        stats.innerHTML = `
                            <div class="stat-item stat-mask">
                                <i class="fas fa-check-circle"></i>
                                <span>已戴口罩: ${maskCount}</span>
                            </div>
                        `;

                        resultContainer.appendChild(stats);
                    }
                }
            };
        };

        // 文件读取错误处理
        reader.onerror = function() {
            if (resultContainer.dataset.currentRequest == requestId) {
                // 显示读取文件错误信息
                resultContainer.innerHTML = `
                    <div class="error-message">
                        <i class="fas fa-exclamation-circle"></i>
                        <p>无法读取图像文件，请尝试其他图片。</p>
                    </div>
                `;
            }
        };

        // 读取文件为DataURL格式
        reader.readAsDataURL(file);
    }

    // ------------- 键盘导航支持 -------------
    dropArea.addEventListener('keydown', (e) => {
        if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault();
            fileInput.click();
        }
    });
});
