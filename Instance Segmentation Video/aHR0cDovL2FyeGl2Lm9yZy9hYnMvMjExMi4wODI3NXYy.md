# SeqFormer: Sequential Transformer for Video Instance Segmentation

Junfeng Wu, Yi Jiang, Song Bai, Wenqing Zhang, and Xiang Bai (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 비디오 인스턴스 분할(Video Instance Segmentation, VIS)이다. VIS는 비디오 내의 객체 인스턴스에 대해 탐지(Detection), 분류(Classification), 분할(Segmentation), 그리고 추적(Tracking)을 동시에 수행해야 하는 매우 도전적인 과제이다.

기존의 VIS 접근 방식은 크게 두 가지로 나뉜다. 첫째는 'Tracking-by-detection' 패러다임으로, 프레임별로 인스턴스를 예측한 뒤 후처리나 별도의 추적 브랜치를 통해 이를 연결하는 방식이다. 하지만 이 방식은 비디오에서 빈번하게 발생하는 가려짐(Occlusion)이나 모션 블러(Motion Blur)에 매우 취약하다는 한계가 있다. 둘째는 비디오 클립이나 전체 비디오를 입력으로 받아 마스크 시퀀스를 직접 예측하는 방식이다. 최근에는 Transformer 구조를 도입한 VisTR이나 IFC 같은 모델들이 등장하였으나, 여전히 효율성과 정확도 사이의 균형을 맞추는 문제와 객체의 움직임에 따른 유연한 어텐션 메커니즘의 부재라는 문제가 남아 있다.

따라서 본 논문의 목표는 별도의 추적 브랜치나 복잡한 후처리 없이도, 비디오 전체에서 인스턴스의 관계를 효과적으로 모델링하여 정확하고 견고한 인스턴스 시퀀스를 예측할 수 있는 end-to-end 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

SeqFormer의 핵심 아이디어는 **'인스턴스 쿼리의 분해(Instance Query Decomposition)'**와 **'프레임 독립적 어텐션'**이다.

연구진은 단일한 인스턴스 쿼리가 비디오 전체의 인스턴스 정보를 캡처하기에 충분하지만, 어텐션 메커니즘은 각 프레임에서 독립적으로 수행되어야 한다고 주장한다. 이는 객체가 비디오 내에서 이동함에 따라 모델이 주목해야 할 공간적 위치가 계속해서 변하기 때문이다. 이를 위해 SeqFormer는 공유된 인스턴스 쿼리를 프레임별 박스 쿼리(Box Query)로 분해하여, 각 프레임의 객체 위치를 정밀하게 추적하고 동시에 비디오 레벨의 강력한 인스턴스 표현을 학습하도록 설계되었다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 다루며 차별점을 제시한다.

1.  **Image Instance Segmentation**: Mask R-CNN과 같은 전통적인 방식부터 CondInst, QueryInst와 같은 최신 one-stage 모델들이 언급된다. 특히 CondInst의 Dynamic Mask Head는 동일 인스턴스가 프레임 간에 동일한 마스크 헤드 파라미터를 공유할 수 있다는 점에서 SeqFormer의 기반이 된다.
2.  **Video Instance Segmentation**: MaskTrack R-CNN, SipMask, STEm-Seg 등의 연구들이 소개된다. 기존 연구들은 프레임 간의 연관성을 찾기 위해 복잡한 추적 헤드나 마스크 전파(Mask Propagation) 방식을 사용하였으나, 이는 계산 비용이 높거나 추적 성능이 불안정하다는 한계가 있다.
3.  **Transformers**: DETR과 Deformable DETR의 성공 이후 VisTR과 IFC가 VIS에 Transformer를 적용하였다. 그러나 VisTR은 비디오 길이에 따라 쿼리 수가 고정되는 문제가 있으며, IFC는 인코더에서 프레임 간 통신을 수행하지만 디코더에서는 여전히 시공간 특징을 평탄화(Flatten)하여 처리한다. SeqFormer는 시공간 차원을 유지하며 프레임별로 독립적인 특징 캡처를 수행함으로써 이들과 차별화된다.

## 🛠️ Methodology

### 전체 아키텍처
SeqFormer는 CNN Backbone, Transformer Encoder, Query Decompose Transformer Decoder, 그리고 세 개의 출력 헤드(Class, Box, Mask Head)로 구성된다.

### 상세 구성 요소 및 절차

**1. Backbone & Transformer Encoder**
입력 비디오 $x_v \in \mathbb{R}^{T \times 3 \times H \times W}$에 대해 CNN 백본이 각 프레임을 독립적으로 처리하여 특징 맵을 추출한다. 이후 $1 \times 1$ 컨볼루션을 통해 채널 차원을 $C=256$으로 맞춘 뒤, Transformer Encoder에서 Deformable Attention을 수행한다. 이때, 시공간 차원을 평탄화하지 않고 그대로 유지하여 각 프레임에 대해 독립적으로 어텐션을 수행한다.

**2. Query Decompose Transformer Decoder**
이 모델의 핵심 부분으로, 비디오 레벨의 Instance Query $I_q$를 프레임별 Box Query $B_t$로 분해하여 처리한다.

- **첫 번째 디코더 층**: 인스턴스 쿼리 $I_q$가 각 프레임의 특징 맵 $f_t$에서 독립적으로 특징을 쿼리하여 프레임별 박스 쿼리를 생성한다.
  $$B_t^1 = \text{DeformAttn}(I_q, f_t)$$
- **이후 디코더 층 ($l > 1$)**: 이전 층의 박스 쿼리 $B_t^{l-1}$를 입력으로 사용하여 위치를 정교화한다.
  $$B_t^l = \text{DeformAttn}(B_t^{l-1}, f_t)$$
- **인스턴스 쿼리 업데이트**: 모든 프레임의 박스 쿼리들을 가중 합(Weighted Sum) 방식으로 결합하여 비디오 레벨의 인스턴스 표현 $I_q^l$를 업데이트한다.
  $$I_q^l = \frac{\sum_{t=1}^T B_t^l \times \text{FC}(B_t^l)}{\sum_{t=1}^T \text{FC}(B_t^l)} + I_q^{l-1}$$
  여기서 $\text{FC}$는 박스 임베딩으로부터 가중치를 학습하는 완전 연결 층이다.

**3. Output Heads**
- **Class Head**: 최종 인스턴스 임베딩 $I_q^{N_d}$를 통해 클래스 확률을 예측한다.
- **Box Head**: 각 프레임의 박스 임베딩 $BE_t$를 3층 FFN에 통과시켜 바운딩 박스의 중심 좌표, 높이, 너비를 예측한다.
- **Mask Head**: Dynamic Convolution을 사용한다. 인스턴스 임베딩 $I_q^{N_d}$로부터 마스크 헤드의 파라미터 $\omega_i$를 생성하며, 이 파라미터는 동일 인스턴스의 모든 프레임에서 공유된다. 최종 마스크는 다음과 같이 생성된다.
  $$\{m_i^t\}_{t=1}^T = \{\text{MaskHead}(F_{\text{mask}}^t, \omega_i)\}_{t=1}^T$$
  여기서 $F_{\text{mask}}^t$는 FPN 스타일의 마스크 브랜치에서 생성된 고해상도 특징 맵과 상대 좌표 맵이 결합된 형태이다.

### 학습 절차 및 손실 함수
모델은 고정된 수의 예측 세트 $N$을 생성하며, 이를 Ground Truth(GT)와 매칭하기 위해 헝가리안 알고리즘(Hungarian Algorithm)을 사용한다. 매칭 비용(Matching Cost)은 다음과 같다.
$$L_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -\hat{p}_{\sigma(i)}(c_i) + L_{\text{box}}(b_i, \hat{b}_{\sigma(i)})$$
최적의 할당 $\hat{\sigma}$가 결정되면, 다음과 같은 헝가리안 손실(Hungarian Loss)을 통해 학습한다.
$$L_{\text{Hung}}(y, \hat{y}) = \sum_{i=1}^N \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \emptyset\}} L_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) + \mathbb{1}_{\{c_i \neq \emptyset\}} L_{\text{mask}}(m_i, \hat{m}_{\hat{\sigma}(i)}) \right]$$
- $L_{\text{box}}$: $L_1$ loss와 Generalized IoU loss의 선형 결합.
- $L_{\text{mask}}$: Dice loss와 Focal loss의 결합.

## 📊 Results

### 실험 설정
- **데이터셋**: YouTube-VIS 2019 및 2021.
- **지표**: Average Precision (AP) 및 Average Recall (AR).
- **백본**: ResNet-50, ResNet-101, ResNeXt-101, Swin-L.
- **학습**: COCO 데이터셋에서 사전 학습 후 YouTube-VIS에서 12 epoch 동안 학습. COCO pseudo videos를 추가하여 오버피팅을 방지하였다.

### 정량적 결과 (YouTube-VIS 2019)
SeqFormer는 동일 백본 대비 기존 SOTA 모델들을 크게 상회하는 성능을 보였다.
- **ResNet-50**: 47.4 AP (기존 SOTA 대비 4.6 AP 향상)
- **ResNet-101**: 49.0 AP (기존 SOTA 대비 4.4 AP 향상)
- **ResNeXt-101**: 51.2 AP (최초로 50 AP 돌파)
- **Swin-L**: 59.3 AP (압도적인 최고 성능 달성)

추론 속도 면에서도 ResNet-50 기준 72.3 FPS를 기록하여, 성능 향상뿐만 아니라 효율성 측면에서도 매우 우수함을 입증하였다.

### 정성적 결과 및 분석
시각화 결과, SeqFormer의 마스크 예측은 시간 흐름에 따라 매우 안정적이며, 가려짐(Occlusion)이나 복잡한 움직임이 있는 상황에서도 객체를 정확하게 추적하는 모습을 보였다.

## 🧠 Insights & Discussion

### 주요 분석 및 강점
1.  **쿼리 분해의 중요성**: 어블레이션 연구 결과, 인스턴스 쿼리를 프레임별 박스 쿼리로 분해하지 않았을 때 성능이 45.1 AP에서 34.1 AP로 급격히 하락하였다. 이는 단일 쿼리를 사용하면 모든 프레임에서 동일한 영역을 샘플링하게 되어, 객체의 움직임을 반영할 수 없기 때문이다.
2.  **시공간 차원 유지**: 특징 맵을 평탄화(Flatten)하여 처리하는 대신 시공간 차원을 유지하고 독립적인 샘플링을 수행했을 때 7.4 AP의 성능 이득을 얻었다. 이는 시간 도메인과 공간 도메인의 서로 다른 특성을 적절히 처리한 결과로 해석된다.
3.  **가중 합(Weighted Sum)의 효용성**: 단순 합(Sum)이나 평균(Average)보다 학습 가능한 가중치를 이용한 합산 방식이 더 높은 성능을 보였다. 이는 객체가 등장하지 않는 프레임의 노이즈를 효과적으로 제거했기 때문이다.

### 한계 및 논의
본 논문은 매우 높은 성능을 달성했지만, 추론 시 전체 비디오를 입력으로 넣는 방식이므로 메모리 제한이 있는 매우 긴 비디오의 경우 클립 단위로 나누어 처리하는 방식(Clip Matching)이 필요함을 언급하고 있다. 또한, Swin Transformer와 같은 강력한 백본에 의존했을 때 성능 향상 폭이 매우 크다는 점은 아키텍처뿐만 아니라 백본의 성능이 VIS 작업에 결정적인 영향을 미침을 시사한다.

## 📌 TL;DR

SeqFormer는 비디오 인스턴스 분할을 위해 **인스턴스 쿼리를 프레임별 박스 쿼리로 분해**하는 새로운 Transformer 구조를 제안한다. 이를 통해 별도의 추적 브랜치나 후처리 없이도 객체의 움직임을 정밀하게 따라가며 마스크 시퀀스를 생성할 수 있다. YouTube-VIS 2019 벤치마크에서 Swin-L 백본 기준 **59.3 AP**라는 압도적인 성능을 기록하였으며, 효율적인 end-to-end 프레임워크로서 향후 VIS 연구의 강력한 베이스라인이 될 가능성이 높다.