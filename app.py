import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# main.py에서 정의한 모델 클래스들을 가져옵니다.
from main import HandwritingCorrectionSystem, Generator, Discriminator

# --- 설정 ---
MODEL_PATH = "hangul_font_model_epoch_5.pth"
NUM_FONTS = 59  # 훈련 시 확인된 폰트 개수
LATENT_DIM = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 모델 로딩 ---
def load_model(model_path):
    """학습된 모델과 가중치를 불러옵니다."""
    print(f"Loading model from {model_path}...")
    
    # 훈련 시와 동일한 구조로 모델을 초기화합니다.
    system = HandwritingCorrectionSystem(
        num_fonts=NUM_FONTS,
        latent_dim=LATENT_DIM,
        device=DEVICE
    )
    
    # 저장된 가중치를 불러옵니다. 옵티마이저는 추론에 필요 없습니다.
    checkpoint = torch.load(model_path, map_location=DEVICE)
    system.generator.load_state_dict(checkpoint['generator_state_dict'])
    system.font_embeddings.load_state_dict(checkpoint['font_embeddings_state_dict'])
    
    system.generator.eval()
    print("Model loaded successfully.")
    return system

# 전역 변수로 모델을 한번만 로드합니다.
correction_system = load_model(MODEL_PATH)

# --- 이미지 변환 및 추론 함수 ---
def generate_correction(drawing, font_id, steps):
    """사용자 입력으로부터 교정 가이드라인을 생성합니다."""
    if drawing is None:
        return []

    # 1. 입력 이미지 전처리
    # Gradio의 Sketchpad는 {'composite': 이미지배열, ...} 형태의 딕셔너리를 반환합니다.
    # 'composite' 키를 사용하여 최종 이미지를 가져옵니다.
    img_array = drawing['composite']
    
    # 모델은 검은 배경에 흰 글씨로 훈련되었으므로 색상 반전이 필요합니다.
    img = Image.fromarray(img_array).convert("L")
    img = Image.fromarray(255 - np.array(img)) # 색상 반전 (흰/검 -> 검/흰)
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 훈련 시와 동일하게 -1~1로 정규화
    ])
    
    tensor_img = transform(img).unsqueeze(0).to(DEVICE)
    
    # 2. 모델 추론
    correction_steps = correction_system.generate_correction_guideline(
        tensor_img, 
        target_font_id=int(font_id),
        num_steps=int(steps)
    )
    
    # 3. 결과 이미지 후처리
    output_images = []
    for step_image in correction_steps:
        # Tensor를 다시 PIL Image로 변환
        img_np = step_image[0].cpu().numpy().squeeze()
        img_np = (img_np + 1) / 2  # 0~1 범위로 되돌리기
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        output_images.append(img_pil)
        
    return output_images

# --- Gradio UI 구성 ---
with gr.Blocks() as demo:
    gr.Markdown("# 한글 손글씨 교정기")
    gr.Markdown("왼쪽 칸에 한글 한 글자를 그리고, 변환하고 싶은 목표 폰트와 교정 단계를 선택한 후 '실행' 버튼을 누르세요.")
    
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(
                label="손글씨를 그려주세요", 
                height=256, 
                width=256,
                image_mode="RGB"
            )
            font_slider = gr.Slider(
                minimum=0, 
                maximum=NUM_FONTS - 1, 
                value=5, 
                step=1, 
                label="목표 폰트 ID"
            )
            steps_slider = gr.Slider(
                minimum=1, 
                maximum=8, 
                value=4, 
                step=1, 
                label="교정 단계 수"
            )
            run_button = gr.Button("실행")
            
        with gr.Column():
            gallery = gr.Gallery(
                label="교정 과정", 
                columns=5, 
                rows=2, 
                object_fit="contain", 
                height="auto"
            )

    run_button.click(
        fn=generate_correction, 
        inputs=[sketchpad, font_slider, steps_slider], 
        outputs=gallery
    )

if __name__ == "__main__":
    demo.launch() 