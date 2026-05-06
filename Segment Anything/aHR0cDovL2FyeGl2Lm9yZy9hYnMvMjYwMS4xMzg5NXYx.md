# OmniOVCD: Streamlining Open-Vocabulary Change Detection with SAM 3

Xu Zhang, Danyang Li, Yingjie Xia, Xiaohang Dong, Hualong Yu, Jianye Wang, Qicheng Li (2026)

## 🧩 Problem to Solve

본 논문은 원격 탐사(Remote Sensing) 분야의 핵심 과제인 변화 탐지(Change Detection, CD)에서 발생하는 폐쇄 집합(Closed-set) 설정의 한계를 해결하고자 한다. 전통적인 CD 방법론은 훈련 데이터에 정의된 특정 카테고리만을 인식할 수 있어, 훈련 과정에서 정의되지 않은 새로운 클래스의 변화를 탐지하지 못하는 문제가 있다.

이를 해결하기 위해 등장한 Open-Vocabulary Change Detection (OVCD)는 자연어 가이드를 통해 임의의 카테고리 변화를 탐지하는 것을 목표로 한다. 그러나 기존의 training-free OVCD 방식들은 CLIP(카테고리 식별)과 DINO(특징 추출) 또는 SAM(세그멘테이션)과 같은 여러 독립적인 모델을 결합하여 사용하는 다단계 파이프라인(Multi-stage pipeline) 구조를 가진다. 이러한 접근 방식은 모델 간의 특징 정렬(Feature alignment) 문제를 야기하며, 단계별로 오차가 누적되어 시스템의 불안정성과 높은 계산 비용을 초래한다는 치명적인 단점이 있다.

따라서 본 논문의 목표는 단일 모델만으로도 효율적이고 안정적으로 작동하며, 높은 일반화 성능을 가진 standalone OVCD 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최신 비전 파운데이션 모델인 **SAM 3 (Segment Anything Model 3)**의 통합된 아키텍처를 활용하여 복잡한 다단계 파이프라인을 단순화하는 것이다. SAM 3는 단일 모델 내에서 세그멘테이션과 식별 능력을 동시에 제공하므로, 외부 모델(CLIP, DINO 등) 없이도 Open-Vocabulary 과업을 수행할 수 있다.

특히, 저자들은 **SFID (Synergistic Fusion to Instance Decoupling)** 전략을 제안한다. 이 전략은 SAM 3의 서로 다른 출력 헤드(Semantic, Instance, Presence)에서 나오는 정보들을 시너지 있게 융합하여 정밀한 시맨틱 맵을 생성하고, 이를 다시 개별 인스턴스 단위로 분리하여 시공간적 일관성을 분석함으로써 가짜 변화(Pseudo-changes)를 억제하고 탐지 정확도를 높이는 설계 구조를 가진다.

## 📎 Related Works

### 1. Segment Anything Model (SAM) 시리즈

SAM은 프롬프트 기반의 제로샷 이미지 세그멘테이션 패러다임을 제시하였다. 이후 SAM 2는 비디오 객체 추적 및 마스크 전파를 위해 메모리 어텐션 메커니즘을 도입하였고, 최신 SAM 3는 Promptable Concept Segmentation (PCS)을 통해 탐지, 세그멘테이션, 추적 기능을 단일 아키텍처로 통합하였다.

### 2. 전통적 및 Open-Vocabulary 변화 탐지

전통적인 CD는 주로 Siamese CNN이나 Transformer 기반 아키텍처를 사용하여 두 시점 이미지의 특징 차이를 분석한다. 하지만 이들은 학습 데이터 외의 카테고리를 인식하지 못한다. 최근의 OVCD 연구인 DynamicEarth 등은 CLIP과 SAM을 결합한 Mask-Compare-Identify (M-C-I) 또는 Identify-Mask-Compare (I-M-C) 방식을 제안하였으나, 앞서 언급한 바와 같이 서로 다른 모델을 결합하는 과정에서 발생하는 불안정성과 정렬 문제라는 한계가 존재한다.

## 🛠️ Methodology

### 전체 파이프라인 구조

OmniOVCD는 입력으로 두 시점의 이미지 $x^{img}_1, x^{img}_2$와 대상 카테고리를 정의하는 텍스트 프롬프트 $x^{txt}$를 받는다. 전체 프로세스는 $\text{SAM 3} \rightarrow \text{SFID (Fusion)} \rightarrow \text{Instance Decoupling} \rightarrow \text{Bi-temporal Matching} \rightarrow \text{Change Mask}$ 순으로 진행된다.

### SFID (Synergistic Fusion to Instance Decoupling) 전략

#### 1. Synergistic Mask Fusion (시너지 마스크 융합)

SAM 3의 세 가지 출력 헤드(Semantic, Instance, Presence)를 결합하여 고정밀 시맨틱 맵을 생성한다.

* **Instance aggregation**: Transformer 기반 디코더에서 생성된 $N_t$개의 인스턴스 쿼리 $P^{inst}_{t}$와 신뢰도 점수 $s^{conf,t}$를 가중치 기반 최대 선택 전략으로 통합한다.
    $$P^{agg}_t(h,w) = \max_{k=1}^{N_t} (P^{(k)}_{inst,t}(h,w) \cdot s^{(k)}_{conf,t})$$
* **Semantic fusion**: 개별 객체 경계에 강한 $P^{agg}_t$와 연속적인 영역 표현에 강한 시맨틱 헤드 출력 $P^{sem}_t$를 픽셀 단위 max-fusion 연산으로 결합한다.
    $$P^{fused,t}(h,w) = \max(P^{sem,t}(h,w), P^{agg,t}(h,w))$$
* **Presence Gating**: 텍스트 쿼리된 개념 $c$가 이미지 내에 실제로 존재하는지 나타내는 Presence head의 점수 $S^{pres,t}$를 곱하여 잘못된 예측(노이즈)을 필터링한다.
    $$P^{(c)final,t} = P^{(c)fused,t} \cdot S^{(c)pres,t}$$
    최종 픽셀 클래스는 $M_t(h,w) = \arg \max_{c \in C} P^{(c)final,t}(h,w)$로 결정된다.

#### 2. Instance Decoupling and Matching (인스턴스 분리 및 매칭)

픽셀 단위의 시맨틱 맵 $M_t$를 객체 단위의 분석으로 전환한다.

* **Decoupling**: 8-connectivity 룰을 적용한 연결 성분 분석(Connected-component analysis)을 통해 연속적인 시맨틱 영역을 개별 인스턴스 집합 $I^t$로 분리한다.
* **Matching**: 두 시점의 인스턴스 간 겹침 비율(Overlap ratio) $R$을 계산한다.
    $$R(I^t_a, I^{t'}_b) = \frac{|I^t_a \cap I^{t'}_b|}{|I^t_a|}$$
* **Change Detection**: $T_1$에서 $T_2$로(Forward), $T_2$에서 $T_1$으로(Backward) 검사를 수행하여, 겹침 비율이 임계값 $\tau_{match}$보다 낮은 인스턴스들을 변화된 영역($C^{T1} \cup C^{T2}$)으로 간주하고 최종 변화 마스크 $M_{change}$를 생성한다.

## 📊 Results

### 실험 설정 및 데이터셋

* **모델**: SAM 3 기반 단일 프레임워크
* **데이터셋**: LEVIR-CD, WHU-CD (건물 변화 탐지), S2Looking (농촌 지역), SECOND (다양한 토지 피복 카테고리)
* **지표**: IoU (Intersection over Union), F1 Score

### 주요 결과

1. **건물 변화 탐지 성능**: LEVIR-CD(67.2), WHU-CD(66.5), S2Looking(24.5)의 IoU를 달성하여 기존의 모든 training-free 방법론을 압도하였다. 특히 DynamicEarth의 다양한 설정보다 훨씬 안정적인 성능을 보였다.
2. **다중 클래스 탐지 (SECOND)**: 건물, 수역, 수목 등 모든 개별 카테고리에서 기존 SOTA 모델보다 높은 IoU를 기록했으며, 클래스 평균 IoU 27.1을 달성하였다.
3. **효율성**: 다중 모델을 결합한 기존 방식보다 GPU 메모리 사용량이 현저히 적고, 추론 속도(Runtime)가 훨씬 빠름을 확인하였다(Fig 4 참조).

### Ablation Study

* **Semantic Head Fusion의 효과**: 시맨틱 헤드 융합을 제거했을 때 성능이 크게 하락하여, 카테고리 수준의 가이드가 인스턴스 분리 단계에 필수적임을 입증하였다.
* **매칭 전략 비교**: 단순 픽셀 비교(PMC)나 Logit-space 거리($L1, L2$) 방식보다 제안한 인스턴스 매칭 방식이 픽셀 노이즈에 강하며 정확한 경계를 유지하는 데 훨씬 효과적임을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 복잡한 모델 조합 없이 SAM 3라는 단일 파운데이션 모델의 내부 구조(Multi-head outputs)를 최적으로 활용함으로써 OVCD의 효율성과 성능을 동시에 잡았다. 특히, 픽셀 단위의 시맨틱 정보와 인스턴스 단위의 기하학적 정보를 결합한 뒤, 다시 인스턴스 레벨에서 시공간적 일관성을 비교하는 **'Fusion $\rightarrow$ Decoupling $\rightarrow$ Matching'** 흐름이 매우 효과적이었음을 알 수 있다.

강점으로는 모델의 단순화(Streamlining)를 통해 추론 속도와 메모리 효율을 극대화하면서도 SOTA 성능을 낸 점을 들 수 있다. 다만, 본 연구는 training-free 방식에 집중하고 있어, 특정 도메인에 특화된 미세 조정(Fine-tuning)이 이루어졌을 때의 성능 향상 폭에 대해서는 명시적으로 다루지 않았다. 또한, 인스턴스 매칭의 임계값 $\tau_{match}$ 설정에 따른 민감도 분석이 추가되었다면 더욱 견고한 분석이 되었을 것이다.

## 📌 TL;DR

OmniOVCD는 SAM 3의 통합 아키텍처를 활용하여, 여러 모델을 복잡하게 연결하던 기존의 Open-Vocabulary 변화 탐지 파이프라인을 단일 모델 기반의 효율적인 구조로 단순화한 프레임워크이다. 특히 SFID 전략을 통해 시맨틱-인스턴스-존재 여부 정보를 융합하고 인스턴스 단위로 변화를 분석함으로써, 4개의 주요 벤치마크에서 SOTA 성능을 달성함과 동시에 추론 속도와 메모리 효율을 크게 개선하였다. 이 연구는 향후 지능형 원격 탐사 분석 시스템의 실용적인 기반이 될 가능성이 높다.
