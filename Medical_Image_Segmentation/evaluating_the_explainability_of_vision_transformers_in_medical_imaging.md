# Evaluating the Explainability of Vision Transformers in Medical Imaging

Leili Barekatain and Ben Glocker (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석 분야에서 Vision Transformer (ViT) 모델들의 성능은 매우 뛰어나지만, 내부의 복잡한 Attention 메커니즘으로 인해 결정 과정에 대한 설명 가능성(Explainability)이 부족하다는 문제를 다룬다. 의료 진단 분야에서는 모델의 예측 결과뿐만 아니라, 그 결과가 임상적으로 유의미한 생물학적 특징에 근거했는지를 확인하는 것이 임상적 신뢰도와 실제 도입에 필수적이다.

따라서 본 연구의 목표는 다양한 ViT 아키텍처와 사전 학습 전략(Pre-training strategies)이 모델의 설명 가능성에 어떤 영향을 미치는지 평가하고, 어떤 조합이 가장 신뢰할 수 있는 시각적 설명을 제공하는지 분석하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 서로 다른 네 가지 ViT 변형 모델과 두 가지 설명 가능성 기법을 체계적으로 비교 분석하여, 의료 영상 작업에서 가장 '충실한(Faithful)' 설명을 제공하는 조합을 찾아낸 것이다. 특히, 단순히 정확도라는 성능 지표에 의존하지 않고, 정량적 지표(Insertion/Deletion AUC)와 정성적 분석(임상적 유의성)을 모두 사용하여 모델의 투명성을 검증하였다. 연구 결과, DINO 기반의 모델과 Grad-CAM의 조합이 가장 국소화되고 신뢰할 수 있는 설명을 제공한다는 점을 밝혀냈다.

## 📎 Related Works

논문은 기존의 ViT 설명 가능성 연구들이 가진 한계를 다음과 같이 지적한다.

- **평가 지표의 부재:** 일부 연구(예: xViTCOS)는 설명 가능성을 정성적인 시각적 정렬(Visual alignment)에만 의존하여 평가했으며, 객관적인 정량적 지표를 제공하지 않았다.
- **아키텍처 탐색의 부족:** 기존의 벤치마크 연구(예: Komorowski et al.)는 기본 ViT 모델의 미세 조정(Fine-tuning)에 국한되었으며, 다양한 ViT 아키텍처나 사전 학습 전략이 설명 가능성에 미치는 영향에 대해서는 충분히 탐구하지 않았다.

본 논문은 이러한 공백을 메우기 위해 ViT, DeiT, DINO, Swin Transformer라는 다양한 구조를 비교군으로 설정하여 차별점을 두었다.

## 🛠️ Methodology

### 1. 평가 대상 모델 (Architectures)

본 연구에서는 다음 네 가지 모델을 사용하였다.

- **ViT (Vision Transformer):** 이미지를 고정 크기의 패치로 나누어 선형 임베딩 후 Transformer Encoder를 통해 전역적 문맥을 캡처한다.
- **DeiT (Data-efficient Image Transformer):** 증류 토큰(Distillation token)을 사용하여 CNN 교사 모델로부터 지식을 학습함으로써, 적은 데이터셋으로도 효율적인 학습이 가능하게 한다.
- **DINO (Self-Distillation with No Labels):** 레이블 없이 교사-학생(Teacher-Student) 구조를 통한 자기 지도 학습(Self-supervised learning)을 수행하여 의미 있는 시각적 표현을 학습한다.
- **Swin Transformer:** 계층적 특징 표현과 Shifted Window Attention을 도입하여 계산 복잡도를 줄이면서도 전역적 문맥을 모델링한다.

### 2. 설명 가능성 기법 (Explainability Techniques)

모델의 결정 근거를 시각화하기 위해 두 가지 접근 방식을 사용하였다.

**A. Attention-Based Method: Gradient Attention Rollout**
기본적인 Attention Rollout은 모든 레이어의 Attention 행렬을 단순히 곱하여 정보의 흐름을 추적한다.
$$\text{rollout} = \hat{A}^{(1)} \cdot \hat{A}^{(2)} \cdot \dots \cdot \hat{A}^{(B)}$$
하지만 이는 예측 클래스와 상관없이 동일한 결과를 낸다는 한계가 있다. 이를 해결하기 위해 **Gradient Attention Rollout**은 Attention 값에 그라디언트(Gradient)를 곱하여, 특정 타겟 클래스의 결정에 긍정적으로 기여한 경로만을 강조함으로써 클래스 특이적인(Class-specific) 맵을 생성한다.

**B. Feature Attribution Method: Grad-CAM**
Grad-CAM은 특정 레이어의 활성화 맵(Activation map)에 대해 타겟 클래스 점수의 그라디언트를 계산한다. 이 그라디언트를 평균 내어 중요도 가중치를 구하고, 이를 활성화 맵과 결합한 후 ReLU 함수를 적용하여 긍정적인 기여를 하는 영역만을 추출한다. 최종적으로 이를 입력 이미지 크기로 업샘플링하여 히트맵을 생성한다.

### 3. 실험 설정

- **데이터셋:** 말초 혈액 세포(PBC) 분류(8개 클래스) 및 유방 초음파 이미지 분류(3개 클래스).
- **학습:** 모든 이미지는 $224 \times 224$로 리사이징되었으며, PyTorch Transformers 라이브러리의 사전 학습된 모델을 미세 조정하였다.

## 📊 Results

### 1. 분류 성능 (Classification Performance)

분류 정확도 면에서는 ViT와 Swin Transformer가 가장 우수한 성능을 보였다.

- **PBC 데이터셋:** ViT (Accuracy: 98.68%, F1: 98.73%)가 가장 높았다.
- **유방 초음파 데이터셋:** Swin (Accuracy: 89.74%, F1: 88.44%)가 가장 높았다.
- DINO는 상대적으로 정확도가 낮게 나타났다 (PBC: 96.97%, 유방 초음파: 80.77%).

### 2. 정량적 설명 가능성 평가

설명 맵의 충실도(Faithfulness)를 측정하기 위해 **Insertion**과 **Deletion** 지표를 사용하였다.

- **Insertion AUC ($\uparrow$):** 중요도가 높은 픽셀부터 추가했을 때 예측 확률이 얼마나 빠르게 상승하는지를 측정한다.
- **Deletion AUC ($\downarrow$):** 중요도가 높은 픽셀부터 제거했을 때 예측 확률이 얼마나 빠르게 하락하는지를 측정한다.

**결과:** 모든 모델에서 **Grad-CAM**이 Gradient Attention Rollout보다 뛰어난 성능을 보였으며, 특히 **DINO** 모델이 Grad-CAM과 결합했을 때 두 데이터셋 모두에서 가장 좋은 AUC 점수를 기록하였다.

### 3. 정성적 분석 및 오류 분석

- **시각적 정밀도:** Grad-CAM은 매우 집중적이고 해석 가능한 히트맵을 생성한 반면, Rollout은 배경 영역까지 포함하여 산만한 결과를 보였다.
- **DINO의 우수성:** DINO + Grad-CAM 조합은 혈액 세포의 핵이나 유방 병변의 경계선 등 임상적으로 유의미한 형태학적 특징을 가장 정확하게 포착하였다.
- **오류 분석:** 모델이 오분류한 사례에서도 Grad-CAM은 모델이 왜 착각했는지를 보여주었다. 예를 들어, Monocyte를 Immature Granulocyte로 오분류한 경우, 모델이 핵의 형태와 과립성을 오해하여 해당 영역에 강하게 반응했음을 확인하였다.

## 🧠 Insights & Discussion

본 연구의 가장 중요한 통찰은 **"최고의 성능(Accuracy)을 가진 모델이 반드시 최고의 설명 가능성(Explainability)을 가지는 것은 아니다"**라는 점이다. ViT와 Swin이 분류 정확도는 더 높았지만, 결정 근거의 명확성과 국소화 능력은 DINO가 압도적이었다.

이는 DINO의 자기 지도 학습(Self-supervised learning) 방식이 정답 레이블에 과적합되지 않고, 이미지 내의 객체 구조와 형태적 특징을 더 본질적으로 학습했기 때문으로 해석할 수 있다. 의료 진단과 같이 안전성이 중요한 도메인에서는 단순히 정확도만으로 모델을 선택할 것이 아니라, 설명 가능성 지표를 함께 고려하여 모델을 선정해야 한다는 강력한 근거를 제시한다.

다만, 본 연구는 사전 학습된 모델을 미세 조정하는 방식에 집중하였으며, 의료 영상에 특화된 완전히 새로운 설명 가능성 구조를 제안한 것은 아니라는 한계가 있다.

## 📌 TL;DR

이 논문은 다양한 ViT 아키텍처(ViT, DeiT, DINO, Swin)와 설명 방법(Grad-CAM, Gradient Attention Rollout)을 의료 영상 작업에서 비교 평가하였다. 실험 결과, **DINO 모델과 Grad-CAM의 조합**이 임상적으로 가장 유의미하고 충실한 시각적 설명을 제공함을 확인하였다. 이는 의료 AI 모델 선정 시 정확도뿐만 아니라 설명 가능성이 핵심적인 기준이 되어야 함을 시사하며, 향후 더 정밀한 의료 전용 XAI 기법 개발의 필요성을 제시한다.
