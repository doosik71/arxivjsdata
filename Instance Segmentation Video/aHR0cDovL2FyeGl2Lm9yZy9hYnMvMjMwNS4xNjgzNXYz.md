# OpenVIS: Open-vocabulary Video Instance Segmentation

Pinxue Guo, Tony Huang, Peiyang He, Xuefeng Liu, Tianjun Xiao, Zhaoyu Chen, Wenqiang Zhang (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 기존의 Video Instance Segmentation (VIS) 모델들이 가진 폐쇄 집합(Closed-set)의 한계이다. 기존 VIS 모델들은 학습 과정에서 정의된 특정 카테고리의 객체들만 탐지하고 세그멘테이션하며 추적할 수 있다. 따라서 학습 데이터에 포함되지 않은 새로운 카테고리의 객체를 인식하기 위해서는 추가적인 어노테이션 데이터 확보와 재학습 과정이 필수적이며, 이는 막대한 시간과 자원 소모를 야기한다.

이러한 한계를 극복하기 위해 저자들은 Open-vocabulary Video Instance Segmentation (OpenVIS)라는 새로운 태스크를 제안한다. OpenVIS의 목표는 학습 단계에서 본 적이 없는 임의의 카테고리에 대해서도 텍스트 설명(카테고리 이름)을 기반으로 비디오 내 객체를 동시에 탐지, 세그멘테이션 및 추적하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 제한된 카테고리의 데이터만으로 가벼운 파인튜닝을 통해 강력한 Open-vocabulary 능력을 갖춘 **InstFormer** 프레임워크를 제안한 것이다. 주요 설계 아이디어는 다음과 같다.

1.  **Open-world Mask Proposal**: 클래스에 구애받지 않고 최대한 많은 잠재적 인스턴스 마스크를 제안하도록 Contrastive Instance Margin Loss를 도입하였다.
2.  **InstCLIP**: 사전 학습된 CLIP을 기반으로 하되, Instance Guidance Attention을 통해 효율적으로 Open-vocabulary 인스턴스 토큰을 생성하여 분류와 추적에 활용한다.
3.  **Universal Rollout Association**: 추적 문제를 다음 프레임의 인스턴스 추적 토큰을 예측하는 문제로 변환함으로써, 특정 카테고리에 종속되지 않는 범용적인 추적 능력을 확보하였다.

## 📎 Related Works

### Video Instance Segmentation (VIS)
기존 VIS는 크게 오프라인(Offline) 방식과 온라인(Online) 방식으로 나뉜다. VisTR이나 Mask2Former-VIS 같은 오프라인 방식은 비디오 전체를 한 번에 처리하여 성능이 높지만, 긴 비디오나 실시간 스트리밍 환경에 적용하기 어렵다. 반면 MaskTrack R-CNN이나 MinVIS 같은 온라인 방식은 프레임별로 마스크를 생성하고 후처리를 통해 추적하므로 유연성이 높다. InstFormer는 온라인 방식의 구조를 채택하여 마스크 제안과 InstCLIP 기반 분류의 유연성을 확보하였다.

### Vision-Language Models (VLMs)
CLIP과 같은 VLM은 방대한 이미지-텍스트 쌍 데이터를 통해 강력한 Zero-shot 인식 능력을 보여주었다. 하지만 이를 VIS에 직접 적용하는 데에는 두 가지 한계가 있다. 첫째, VLM은 일반 이미지로 학습되었기에 마스킹 처리된 이미지 입력 시 성능이 하락하는 도메인 갭이 존재한다. 둘째, 프레임 내 $N$개의 인스턴스마다 VLM의 Vision Encoder를 $N$번 실행하는 것은 연산 비용 측면에서 매우 비효율적이다.

### Open-Vocabulary Segmentation
최근 OVSeg 관련 연구들은 클래스 없는 제안(Class-agnostic proposal)을 먼저 추출하고, 이후 텍스트 특징과의 유사도를 계산하는 2단계 프레임워크를 주로 사용한다. InstFormer는 이와 유사한 흐름을 따르지만, 효율적인 인스턴스 토큰 임베딩을 통해 연산 비용을 줄이고 비디오 수준의 인스턴스 연관(Association) 기능을 통합했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

InstFormer는 마스크 제안, 인스턴스 표현, 인스턴스 연관의 세 단계 파이프라인으로 구성된다.

### 1. Open-world Mask Proposal
사용자가 어떤 객체를 선택할지 알 수 없으므로, 가능한 한 많은 서로 다른 인스턴스를 제안하는 것이 중요하다. 본 연구에서는 Mask2Former를 기반으로 하되, 쿼리들이 동일한 인스턴스에 중복 할당되는 것을 방지하기 위해 **Contrastive Instance Margin Loss** ($L_{SC}$)를 도입하였다.

$$L_{SC} = \sum_{i=0}^N \sum_{j=0}^N \max(0, \cos(Q_i^t, Q_j^t) - \alpha)$$

여기서 $\cos(\cdot, \cdot)$는 코사인 유사도이며, $\alpha$는 마진 값이다. 이 손실 함수는 너무 유사한 쿼리 토큰들을 서로 밀어내어, 쿼리들이 서로 다른 객체 인스턴스에 다양하게 할당되도록 유도한다.

### 2. Open-vocabulary Instance Representation (InstCLIP)
마스킹된 이미지를 반복적으로 CLIP에 입력하는 비효율성을 해결하기 위해 **InstCLIP**을 제안한다. InstCLIP은 CLIP의 Vision Transformer를 수정하여 **Instance Guidance Attention** 레이어를 적용한 구조이다.

- **Instance Guidance Attention**: 마스크 제안 네트워크에서 생성된 로그잇(logits) 값을 기반으로 어텐션 마스크 $M$을 생성한다. 이를 통해 단 한 번의 순전파(Forward pass)로 $N$개의 인스턴스 토큰이 각각 서로 다른 영역에 집중하도록 가이드한다.
- **수식**:
  $$X_t^l = \text{softmax}(W_q X_{t}^{l-1} \cdot W_k X_{t}^{l-1} + M) \cdot W_v X_{t}^{l-1}$$
- **구성 요소**: 입력 $X_t^{l-1}$은 이미지 패치 토큰($V_t$), 학습 가능한 인스턴스 토큰($I_l$), 그리고 저정보 특징을 수집하여 어텐션 맵을 깨끗하게 만드는 레지스터 토큰($R_l$)으로 구성된다.
- **분류**: 최종 생성된 인스턴스 토큰 $I_t^L$과 CLIP 텍스트 인코더에서 추출한 단어 임베딩 $E$ 간의 유사도를 계산하여 카테고리를 결정한다.
  $$C_t = \text{argmax}(\text{softmax}(I_t^L \cdot E^\top))$$

학습 시에는 CLIP의 가중치를 대부분 동결하고, LoRA(Low-Rank Adaptation)를 통해 쿼리($Q$)와 밸류($V$) 투영 층만 가볍게 파인튜닝한다.

### 3. Universal Rollout Association
특정 카테고리에 최적화된 추적기는 Open-vocabulary 상황에서 성능이 저하된다. 이를 해결하기 위해 추적 문제를 '다음 프레임의 추적 토큰을 예측하는 문제'로 재정의한 **Universal Rollout Association**을 제안한다.

- **추적 토큰 구성**: InstCLIP의 인스턴스 토큰($I_t$)과 마스크 제안 네트워크의 쿼리($Q_t$)를 결합하여 범용 추적 토큰 $T_{Tr}^t$를 생성한다.
  $$T_{Tr}^t = \text{Concat}(I_t, Q_t)$$
- **Rollout Tracker**: 단순한 RNN 레이어를 사용하여 이전 프레임들의 정보를 바탕으로 다음 프레임의 예측 토큰 $T_A^t$를 생성한다.
  $$T_A^t = \text{RNN}(T_{Tr}^{t-1}, h_{t-1})$$
- **연관 절차**: 예측된 토큰 $T_A^t$와 실제 토큰 $T_{Tr}^t$ 사이의 코사인 유사도를 계산하고, 헝가리안 매칭(Hungarian matching)을 통해 인스턴스를 연결한다.
- **학습 목표**: 예측 토큰이 실제 토큰과 일치하도록 하는 예측 추적 손실 함수 $L_T$를 사용한다.
  $$L_T = \sum_{i=1}^N \sum_{j=1}^N \text{CE}(\cos(T_A^t(i), T_{Tr}^t(j)), 1_{[i=j]})$$

## 📊 Results

### 실험 설정
- **데이터셋**: YouTube-VIS(학습, 40개 카테고리), BURST(테스트, 482개 카테고리), UVO(Open-world 제안 능력 평가), LVVIS(Novel 카테고리 평가)를 사용하였다.
- **평가 지표**: 비디오 레벨의 Average Precision (AP) 및 Average Recall (AR)을 사용하였다.
- **비교 대상**: Fully-supervised 방식(MinVIS, BoxTracker 등)과 Open-vocabulary 방식(Detic-SORT, OV2Seg 등)을 비교 대상으로 설정하였다.

### 주요 결과
1.  **전체 성능**: BURST 데이터셋에서 InstFormer는 4.2 AP를 기록하며, 기존 Open-vocabulary 베이스라인인 OV2Seg(3.7 AP)와 Fully-supervised 방식들을 큰 격차로 앞섰다.
2.  **Zero-shot 능력**: 학습 시 보지 못한 카테고리만 포함된 BURST-uncommon(404개 카테고리)에서 OV2Seg 대비 45% 향상된 성능을 보였으며, LVVIS-novel에서도 최상위 성능을 달성하였다. 이는 InstCLIP이 CLIP의 Zero-shot 능력을 잘 유지하고 있음을 입증한다.
3.  **Open-world 제안 능력**: UVO 데이터셋 평가 결과, Contrastive Instance Margin Loss를 적용한 마스크 제안 네트워크가 기존 방식보다 더 정교하고 다양한 인스턴스를 제안함을 확인하였다.
4.  **Fully-supervised VIS 성능**: YouTube-VIS 데이터셋에서 51.8 AP를 기록하여, Open-vocabulary 모델임에도 불구하고 최신 Fully-supervised 모델들과 경쟁 가능한 수준의 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석
InstFormer의 가장 큰 성과는 VLM의 일반적인 제로샷 인식 능력과 비디오 인스턴스 레벨의 특수성을 성공적으로 결합했다는 점이다. 특히, 추적 과정을 카테고리 독립적인 '토큰 예측' 문제로 변환함으로써, 학습 데이터에 없는 객체라도 특징 공간에서의 연속성만 있다면 추적할 수 있게 만든 설계가 매우 효율적이다. 또한, 레지스터 토큰의 도입과 정밀한 마진 $\alpha$ 설정이 성능 향상에 핵심적인 역할을 했음이 어블레이션 연구를 통해 밝혀졌다.

### 한계 및 논의사항
논문에서는 가벼운 파인튜닝만으로 높은 성능을 냈다고 주장하지만, 여전히 CLIP이라는 거대 모델에 의존하고 있어 추론 시의 실시간성 확보에 대한 구체적인 FPS 수치나 연산량 분석이 부족하다. 또한, 텍스트 프롬프트 구성(14가지 앙상블 사용)에 따라 Zero-shot 성능이 민감하게 변할 가능성이 있으나 이에 대한 분석은 명시되지 않았다.

## 📌 TL;DR

본 논문은 학습되지 않은 임의의 객체를 비디오에서 추적하고 세그멘테이션하는 **Open-vocabulary Video Instance Segmentation (OpenVIS)** 태스크를 위한 **InstFormer**를 제안한다. 다양성을 높인 마스크 제안 네트워크, 효율적인 인스턴스 토큰 추출기인 **InstCLIP**, 그리고 카테고리 무관하게 작동하는 **Universal Rollout Tracker**를 통해 기존 모델보다 월등한 제로샷 인식 및 추적 성능을 달성하였다. 이 연구는 향후 특정 도메인에 국한되지 않는 범용 비디오 분석 시스템 구축에 중요한 기여를 할 것으로 보인다.