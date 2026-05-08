# A Fast and Efficient Modern BERT based Text-Conditioned Diffusion Model for Medical Image Segmentation

Venkata Siddharth Dhara and Pawan Kumar (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 질병 진단 및 치료 계획 수립에 필수적인 과정이지만, 모델 학습을 위해 필요한 픽셀 단위의 정밀한 주석(pixel-wise annotations)을 생성하는 데 막대한 비용과 시간이 소요된다. 특히 이러한 데이터셋 구축에는 방사선 전문의나 병리학자와 같은 전문가의 도메인 지식이 필수적이어서, 실제 임상 환경에서 모델의 확장성과 실용적인 구현에 큰 병목 현상이 되고 있다.

본 논문의 목표는 이러한 레이블 효율성(label-efficiency) 문제를 해결하기 위해, 임상 현장에서 쉽게 얻을 수 있는 텍스트 형태의 진단 보고서나 주석을 활용하는 것이다. 저자들은 텍스트 정보를 조건부 입력으로 사용하는 확산 모델(Diffusion Model)을 통해, 적은 양의 픽셀 레이블만으로도 정교한 분할 성능을 내면서 동시에 학습 및 추론 속도를 개선한 **FastTextDiff** 프레임워크를 제안한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 기존의 텍스트 조건부 분할 모델인 TextDiff의 텍스트 인코더를 최신 언어 모델인 **ModernBERT**로 교체하여 연산 효율성과 시맨틱 표현 능력을 동시에 높이는 것이다.

ModernBERT는 2조 개의 토큰으로 학습되었으며, 특히 의료 데이터셋인 MIMIC-III 및 MIMIC-IV를 통해 추가 학습되어 긴 의료 텍스트 시퀀스를 효율적으로 처리할 수 있다. 또한, Flash Attention 2와 교차 주의 집중(Alternating Attention) 메커니즘을 통해 메모리 사용량을 줄이고 학습 속도를 획기적으로 향상시켰다. 이를 통해 텍스트의 시맨틱 정보와 이미지의 시각적 특징을 효과적으로 결합하여 레이블이 부족한 상황에서도 높은 분할 정확도를 달성하고자 하였다.

## 📎 Related Works

의료 영상 분할 분야에서는 U-Net과 같은 CNN 기반 모델과 TransUNet, Swin-UNet 같은 Transformer 기반 모델이 널리 사용되어 왔다. 그러나 이들은 모두 방대한 양의 정밀한 주석 데이터셋에 의존한다는 한계가 있다. 최근에는 생성 모델인 Denoising Diffusion Probabilistic Models(DPMs)가 이미지 합성뿐만 아니라 특징 추출기로서 분할 작업에 활용되기 시작했다.

비전-언어 모델(VLMs)인 GLoRIA나 LViT 등은 이미지와 텍스트의 공동 표현(joint representations)을 학습하여 의료 영상 분석에 적용되었으나, 확산 모델의 계층적 특징 공간을 직접적으로 활용하는 방식은 드물었다. 특히 기존의 TextDiff 모델은 Clinical BioBERT를 텍스트 인코더로 사용하여 가능성을 보여주었으나, 긴 시퀀스 처리 효율과 학습 속도 면에서 최적화가 부족했다는 점이 본 연구의 차별점이다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

FastTextDiff는 크게 **이미지 인코더(Image Encoder)**와 **텍스트 인코더(Text Encoder)** 두 가지 브랜치로 구성되며, 이들은 **교차 모달 주의 집중(Cross-modal Attention)** 모듈을 통해 통합된다.

1. **이미지 인코더**: 사전 학습된 U-Net 기반의 확산 모델을 사용하여 입력 이미지 $x$에 노이즈가 섞인 버전 $x_t$로부터 중간 시각적 특징 $h$를 추출한다.
2. **텍스트 인코더**: ModernBERT를 사용하여 진단 텍스트 $t$를 문맥적 텍스트 임베딩 $\tilde{t}$로 변환한다.
3. **통합 및 출력**: U-Net 디코더 블록 내에 삽입된 교차 모달 주의 집중 모듈이 시각적 특징 $h$와 텍스트 임베딩 $\tilde{t}$를 융합하며, 최종적으로 분할 마스크 $\hat{x}$를 생성한다.

특이사항으로, 학습 시에는 오직 **교차 모달 블록(cross-modal block)**과 **픽셀 분류기 블록(pixel classifier block)**만 학습 가능하도록 설정하고, 나머지 블록은 동결(frozen)시켜 효율성을 높였다.

### 주요 메커니즘 및 방정식

#### 1. 확산 모델 (Diffusion Models)

이미지에 점진적으로 가우시안 노이즈를 추가하는 순방향 과정(Forward Process)과 이를 다시 복원하는 역방향 과정(Backward Process)을 거친다. 역방향 과정의 핵심은 노이즈 제거 네트워크 $\epsilon_\theta$가 추가된 노이즈 $\epsilon$을 예측하는 것이다.
$$\epsilon_\theta(x_t, t) \approx \epsilon$$

#### 2. 교차 모달 주의 집중 (Cross-modal Attention)

이미지 특징 $h$와 텍스트 특징 $\tilde{t}$를 정렬하기 위해 Scaled Dot-Product Attention을 사용한다.
$$H_{z,t} = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
여기서 쿼리(Query) $Q = hW_q$는 이미지 특징에서, 키(Key) $K = \tilde{t}W_k$와 밸류(Value) $V = \tilde{t}W_v$는 텍스트 특징에서 유도된다. 이를 통해 텍스트 정보에 의해 변조된 강화된 이미지 특징 표현 $H_{z,t}$를 얻는다.

#### 3. 손실 함수 (Loss Function)

공간적 겹침 정확도(Dice Loss)와 픽셀 단위 분류 성능(Cross-Entropy Loss)을 모두 잡기 위해 두 손실 함수를 결합하여 사용한다.
$$L = L_{\text{Dice}} + L_{\text{CE}}$$

## 📊 Results

### 실험 설정

- **데이터셋**: MoNuSeg (병리 이미지), QaTa-COV19 (흉부 X-ray), MosMedData+ (흉부 CT)의 3가지 데이터셋을 사용하였다.
- **지표**: Intersection over Union (IoU)와 Dice Similarity Coefficient (DSC)를 사용하여 성능을 측정하였다.
- **환경**: NVIDIA RTX 4090 GPU 1대를 사용하여 50 epoch 동안 학습하였다.

### 주요 결과

1. **분할 성능**: FastTextDiff는 MoNuSeg와 QaTa-COV19 데이터셋에서 UNet, TransUNet, SwinUNet 등 기존 SOTA 모델들과 경쟁하거나 더 우수한 성능을 보였다. 특히 TransUNet(93.19M 파라미터)에 비해 훨씬 적은 파라미터(9.68M)만으로도 높은 성능을 달성하였다.
2. **텍스트 인코더 비교**: ModernBERT를 사용했을 때 기본 BERT나 Clinical BioBERT를 사용한 TextDiff보다 전반적으로 높은 Dice/IoU 점수를 기록하였다. 특히 512의 긴 컨텍스트 길이를 사용할 때 성능이 향상되는 경향을 보였다.
3. **효율성**: ModernBERT의 Flash Attention 2 덕분에 학습 및 추론 속도가 획기적으로 개선되었다. Table 3에 따르면, 모든 데이터셋에서 TextDiff 대비 학습 시간과 이미지당 추론 시간이 유의미하게 단축되었다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 최신 언어 모델인 ModernBERT가 제공하는 정교한 시맨틱 가이드가 확산 모델의 경량화된 백본으로도 충분히 고성능의 분할 결과를 낼 수 있음을 입증하였다. 이는 시각적 정보가 모호한 상황에서 텍스트 정보가 보완적인 역할을 수행하여, 매우 복잡한 시각 전용 특징 추출기 없이도 정확한 영역 지정이 가능함을 시사한다. 또한, 연산 효율성을 극대화하여 리소스가 제한된 임상 환경에서도 배포 가능성을 높였다는 점이 긍정적이다.

### 한계 및 비판적 해석

일부 데이터셋(MosMedData+)에서는 기존 TextDiff보다 성능이 약간 낮게 나타났다. 이는 ModernBERT가 학습된 MIMIC-IV와 기존 모델의 MIMIC-III 데이터 간의 특성 차이, 혹은 해당 데이터셋의 특성상 Clinical BioBERT가 더 적합한 특징을 포착했을 가능성이 있다.

또한, 본 연구는 주로 지도 학습 기반의 성능 향상에 집중하고 있으나, 저자들이 언급했듯이 텍스트 설명과 레이블 없는 이미지만을 사용하는 약지도 학습(weakly-supervised)이나 준지도 학습(semi-supervised) 설정으로 확장한다면 레이블 부족 문제를 더욱 근본적으로 해결할 수 있을 것이다.

## 📌 TL;DR

**FastTextDiff**는 ModernBERT를 텍스트 인코더로 채택한 텍스트 조건부 확산 모델로, 의료 영상 분할에서 레이블 효율성과 연산 속도를 동시에 개선한 프레임워크이다. ModernBERT의 효율적인 아키텍처와 교차 모달 주의 집중 메커니즘을 통해, 적은 파라미터로도 기존 SOTA 모델들과 대등하거나 우수한 분할 성능을 내면서 학습 및 추론 시간을 크게 단축시켰다. 이 연구는 다중 모달 학습이 의료 영상 분석의 데이터 부족 문제를 해결하는 실질적인 대안이 될 수 있음을 보여준다.
