# Consistent Video Instance Segmentation with Inter-Frame Recurrent Attention

Quanzeng You, Jiang Wang, Peng Chu, Andre Abrantes, Zicheng Liu (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Video Instance Segmentation(VIS)에서 발생하는 **시간적 일관성(Temporal Consistency)의 부족** 문제이다. VIS는 각 프레임에서 객체의 세그멘테이션 마스크를 예측함과 동시에 여러 프레임에 걸쳐 동일한 객체를 동일한 ID로 연결(Association)하는 것을 목표로 한다.

최근의 end-to-end 방식의 VIS 모델들은 병렬 시퀀스 디코딩 프레임워크를 사용하여 높은 마스크 품질을 보여주지만, 인접한 프레임 간의 시간적 일관성을 명시적으로 모델링하지 않는다. 이로 인해 객체들이 서로 겹치거나 동일한 카테고리의 객체가 여러 개 등장하는 까다로운 상황에서 다음과 같은 문제가 발생한다:
- **Instance ID change**: 동일 객체의 ID가 갑자기 변경됨
- **Instance ID switch**: 두 객체의 ID가 서로 바뀜
- **Instance ID merged**: 서로 다른 객체가 하나의 ID로 합쳐짐
- **Instance lost**: 추적하던 객체가 갑자기 사라짐

따라서 본 논문의 목표는 인접 프레임 간의 상관관계와 전역적 시간 문맥(Global Temporal Context)을 모두 모델링하여 시간적으로 일관된 VIS 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Inter-Frame Recurrent (IFR) Attention** 프레임워크를 도입하는 것이다. 이 설계의 중심 직관은 객체의 외형과 위치가 인접 프레임 간에 매우 유사하다는 점을 이용하여, 이전 프레임의 정보를 현재 프레임의 예측에 명시적으로 전달하는 것이다.

이를 위해 저자들은 쿼리를 두 가지 수준으로 분리하였다:
1. **Clip-level queries**: 비디오 클립 전체의 전역적 정보를 요약하고 객체의 카테고리를 예측한다.
2. **Frame-level queries**: 각 프레임의 구체적인 정보를 모델링하여 해당 프레임의 세그멘테이션 마스크를 예측한다.

이 두 쿼리를 IFR attention 모듈을 통해 반복적으로 업데이트함으로써, 개별 프레임의 외형, 인접 프레임 간의 시간적 상관관계, 그리고 클립 수준의 전역 문맥을 동시에 학습하도록 설계하였다.

## 📎 Related Works

기존의 VIS 접근 방식은 크게 두 가지로 나뉜다:
1. **Per-frame methods (Tracking-by-detection)**: 각 프레임을 독립적으로 세그멘테이션한 후 추적 알고리즘을 통해 연결하는 방식이다. 효율적이지만 전역적인 문맥 활용 능력이 떨어진다.
2. **End-to-end per-clip methods**: 비디오 클립 전체를 입력으로 하여 한 번에 마스크와 연결 관계를 예측하는 방식이다. VisTR, IFC, Mask2Former 등이 이에 해당하며, 일반적으로 정확도가 더 높다.

본 논문에서 지적하는 기존 end-to-end 방식(특히 Mask2Former 등)의 한계는 전역 쿼리만을 사용하여 마스크를 생성하기 때문에, 인접 프레임 간의 관계를 명시적으로 고려하지 않아 시간적 일관성이 떨어진다는 점이다. IFR framework는 이러한 한계를 극복하기 위해 recurrent 구조의 attention을 도입하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
전체 파이프라인은 **Backbone $\rightarrow$ Multi-scale Transformer Encoder $\rightarrow$ Inter-frame Recurrent Transformer Decoder** 순으로 구성된다.
- **Backbone**: 이미지 기반 백본을 사용하여 프레임별 외형 특징(Feature map)을 추출한다.
- **Multi-scale Transformer Encoder**: Deformable Transformer를 사용하여 다양한 스케일의 특징 맵을 융합한다.
- **IFR Transformer Decoder**: 본 논문의 핵심으로, clip-level query와 frame-level query를 상호작용시켜 최종 마스크와 클래스를 예측한다.

### Inter-Frame Recurrent (IFR) Attention
$i$번째 디코더 레이어에서 clip-level query를 $Q^i$, $t$번째 프레임의 frame-level query를 $Q^t_i$라고 할 때, 동작 과정은 다음과 같다.

1. **Cross-Attention**: 현재 프레임의 특징 맵 $F_t$를 기반으로, 이전 프레임의 query $Q^{t-1}_i$와 현재 클립 쿼리 $Q^i$를 업데이트한다.
   $$Q'^t_i = [\text{Attn}(Q^{t-1}_i, F_t), \text{Attn}(Q^i, F_t)]$$
   (여기서 $\text{Attn}$은 표준적인 Multi-head attention을 의미하며, 수식 (1)과 같이 Softmax 기반으로 계산된다.)

2. **Self-Attention**: 업데이트된 두 그룹의 쿼리를 결합하여 self-attention을 수행한 후, 다시 분리하여 현재 프레임의 최종 쿼리 $Q^t_i$를 얻는다.
   $$Q^t_i = \text{Chunk}(\text{Attn}(Q'^t_i, Q'^t_i))$$

3. **Clip-level Update**: 모든 프레임의 frame-level query $\{Q^1_i, \dots, Q^n_i\}$를 Temporal Cross-Attention을 통해 집계하여 다음 레이어의 클립 쿼리 $Q^{i+1}$를 생성한다.

### 손실 함수 (Training Loss)
모델은 Hungarian algorithm을 통해 ground-truth 인스턴스와 쿼리를 매칭하며, 전체 손실 함수 $L$은 다음과 같이 정의된다:
$$L = \sum_{i=1}^N \lambda_c L_{cls}(c_i, \hat{p}_{\pi_i}) + \mathbb{1}_{c_{\pi_i} \neq \emptyset} [\lambda_m \bar{L}_m(m_i, \hat{m}_{\pi_i}) + \lambda_D \bar{L}_D(m_i, \hat{m}_{\pi_i})]$$

여기서 $L_{cls}$는 Focal loss를 사용하며, $\bar{L}_m$과 $\bar{L}_D$는 각각 마스크 cross-entropy 손실과 DICE 손실이다. 특히, 본 논문은 **Auxiliary cross-frame mask prediction loss**를 제안한다. 이는 특정 프레임의 쿼리가 동일 인스턴스의 다른 프레임 마스크까지 예측하도록 강제함으로써 시간적 정보 전파를 촉진한다:
$$\bar{L}_m(m_i, \hat{m}_{\pi_i}) = \sum_{t_1} [L_m(m^{t_1}_i, \hat{m}^{t_1}_{\pi_i}) + \lambda_e \sum_{t_2 \neq t_1} L_m(m^{t_2}_i, \hat{m}^{t_1}_{\pi_i})]$$
$\lambda_e$는 cross-frame loss의 가중치이며, 실험을 통해 0.3이 최적임이 확인되었다.

### Test Time Augmentation (TTA)
추론 단계에서 입력 해상도를 다양하게 하여 여러 결과 $\{R_1, \dots, R_n\}$을 얻은 뒤, 하나의 결과 $R_p$를 기준(pivot)으로 삼아 다른 결과들을 매칭하고 평균을 내는 방식으로 정확도와 강건성을 높였다.

## 📊 Results

### 실험 설정
- **데이터셋**: YouTubeVIS-2019, YouTubeVIS-2021
- **백본**: ResNet-50, ResNet-101, Swin-T, Swin-S, Swin-B, Swin-L
- **지표**: AP (Average Precision), $\text{AP}_{50}$, $\text{AP}_{75}$, AR (Average Recall) 및 MOT(Multiple Object Tracking) 지표($\text{IDF1}$, $\text{MOTA}$, $\text{IDs}$ 등)

### 정량적 결과
- **정확도**: YouTubeVIS-2019에서 Swin-L 백본 기준 $62.1\%$ AP를 달성하여 기존 SOTA(Mask2Former) 대비 약 $2\%$ 향상되었다. TTA 적용 시 $62.6\%$까지 상승한다.
- **시간적 일관성**: Table 3의 MOT 지표 분석 결과, $\text{IDF1}$과 $\text{MOTA}$가 크게 향상되었고, 특히 $\text{IDs}$(Identity Switches) 수치가 Mask2Former 대비 비약적으로 감소하였다 (예: YTVIS19 Swin-L 기준 218 $\rightarrow$ 26). 이는 제안 방법이 ID 유지 능력이 훨씬 뛰어남을 입증한다.

### 정성적 결과
시각화 결과(Figure 5), Mask2Former가 동일 객체에 대해 중복 예측을 하거나 ID를 혼동(switch)하는 반면, 제안 모델은 복잡한 움직임이 있는 시나리오에서도 단일하고 일관된 ID를 부여하는 모습을 보였다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 연구는 end-to-end VIS 모델이 가진 고질적인 문제인 '시간적 일관성 부족'을 recurrent attention 구조와 보조 손실 함수라는 비교적 단순하면서도 효과적인 방법으로 해결하였다. 특히, 전역 쿼리(Clip-level)와 지역 쿼리(Frame-level)를 분리하여 상호작용시킨 설계가 전역 문맥 유지와 지역적 연속성 확보라는 두 마리 토끼를 모두 잡은 것으로 평가된다.

### 한계 및 논의
1. **계산 복잡도**: 저자들은 모델이 실시간 추론을 하기에는 계산 비용이 너무 높다고 명시하였다. Per-clip 추론을 구현하더라도 실시간 요구사항을 충족하기 어렵다는 점이 한계로 지적된다.
2. **백본 단계의 융합**: 현재의 시간적 정보 융합은 네트워크의 Head 부분에서 주로 이루어진다. 향후 연구에서는 백본 단계에서부터 시간적 정보를 더 효과적으로 집계하는 방법이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 end-to-end 비디오 인스턴스 세그멘테이션에서 발생하는 ID 스위칭 및 소실 문제를 해결하기 위해 **Inter-Frame Recurrent (IFR) Attention**과 **Cross-frame mask prediction loss**를 제안한다. 이 방법은 인접 프레임 간의 정보를 recurrent하게 전달하여 시간적 일관성을 획기적으로 개선하였으며, YouTubeVIS-2019 및 2021 데이터셋에서 SOTA 성능을 달성하였다. 이는 향후 고품질의 비디오 객체 추적 및 세그멘테이션 연구에 있어 시간적 연속성 모델링의 중요성을 시사한다.