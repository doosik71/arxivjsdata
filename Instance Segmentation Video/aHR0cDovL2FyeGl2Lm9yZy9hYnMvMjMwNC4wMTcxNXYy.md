# Towards Open-Vocabulary Video Instance Segmentation

Haochen Wang, Cilin Yan, Shuai Wang, Xiaolong Jiang, Xu Tang, Yao Hu, Weidi Xie, Efstratios Gavves (2023)

## 🧩 Problem to Solve

기존의 Video Instance Segmentation (VIS) 연구들은 훈련 단계에서 정의된 닫힌 집합(closed set)의 카테고리에 대해서만 객체를 분할하고 분류하는 성능에 집중해 왔다. 이러한 폐쇄적 어휘(closed-vocabulary) 패러다임은 실제 환경에서 훈련 시 보지 못한 새로운 카테고리의 객체를 마주했을 때 이를 처리할 수 없는 일반화 능력의 한계를 가진다.

반면, 최근의 Open-World Tracking (OWT) 연구들은 클래스에 구애받지 않고 모든 객체를 분할하고 추적하는 것을 목표로 하지만, 비디오 수준의 작업(예: 비디오 캡셔닝, 행동 인식)에 필수적인 '객체 분류' 능력이 결여되어 있다. 따라서 본 논문은 훈련 시 보지 못한 새로운 카테고리를 포함하여 임의의 오픈셋 카테고리에 대해 객체를 동시에 분할(segment), 추적(track), 그리고 분류(classify)하는 **Open-Vocabulary Video Instance Segmentation**이라는 새로운 과제를 정의하고 이를 해결하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음 세 가지로 요약된다.

1. **Open-Vocabulary VIS 과제 정의**: 전통적인 VIS를 확장하여, 훈련 데이터에 없는 새로운 카테고리에 대해서도 분할, 추적, 분류를 동시에 수행하는 새로운 태스크를 제안하였다.
2. **LV-VIS 데이터셋 구축**: Open-Vocabulary VIS를 벤치마킹하기 위해 1,196개의 다양한 카테고리와 4,828개의 비디오, 544k 개 이상의 인스턴스 마스크를 포함하는 Large-Vocabulary Video Instance Segmentation (LV-VIS) 데이터셋을 수집하였다. 이는 기존 데이터셋보다 카테고리 규모가 한 자릿수 이상 크다.
3. **OV2Seg 아키텍처 제안**: Memory-Induced Transformer 구조를 기반으로 한 end-to-end 모델인 OV2Seg를 제안하였다. 이 모델은 복잡한 제안-감소-결합(propose-reduce-association) 과정을 단순화하여 실시간에 가까운 추론 속도로 Open-Vocabulary VIS를 수행한다.

## 📎 Related Works

본 논문에서 다루는 관련 연구와 기존 방식의 한계는 다음과 같다.

- **Video Instance Segmentation**: 프레임 기반 및 클립 기반 방법론으로 나뉘며, 대규모의 비디오 데이터셋에서 훈련되어야 한다는 제약이 있다. 특히 훈련된 고정 카테고리 외의 객체는 처리하지 못한다.
- **Open-Vocabulary Object Detection**: CLIP과 같은 사전 학습된 시각-언어 모델의 지식을 증류(distillation)하거나 자가 학습(self-training)을 통해 어휘를 확장한다. 하지만 대부분 2단계 검출기 구조를 따르며, 복잡한 파이프라인으로 인해 추론 속도가 느리다는 단점이 있다.
- **Open-World Tracking**: 모든 가시적 객체를 추적하는 데 집중하지만, 객체가 어떤 클래스인지 분류하는 능력은 부족하다.

OV2Seg는 이러한 한계들을 극복하기 위해, 이미지 수준의 오픈 보캐블러리 검출 능력과 비디오 수준의 장기 기억 추적 능력을 end-to-end 구조로 통합하여 효율성과 일반화 성능을 동시에 확보하였다.

## 🛠️ Methodology

OV2Seg는 크게 세 가지 모듈로 구성된다.

### 1. Universal Object Proposal
모든 카테고리의 객체를 효율적으로 제안하기 위해 클래스 독립적(class-independent) 쿼리를 사용한다. 
- 입력 프레임 $I_t$는 Transformer Encoder $\Phi^{ENC}$를 통해 멀티스케일 특징 맵 $F$를 생성한다.
- $N$개의 학습 가능한 클래스 독립적 쿼리 $Q^I \in \mathbb{R}^{N \times d}$가 Transformer Decoder $\Phi^{DEC}$에 입력되어 객체 중심 쿼리 $Q$를 생성한다:
  $$Q = \Phi^{DEC}(F, Q^I) \in \mathbb{R}^{N \times d}$$
- 생성된 $Q$는 Mask Head $H_m$과 Object Score Head $H_o$를 통해 각각 세그멘테이션 마스크 $m$과 객체 존재 확률 $s^{obj}$를 출력한다.

### 2. Memory-Induced Tracking
연속된 프레임 간의 객체 일관성을 유지하고 장기 의존성을 확보하기 위해 Memory Queries $Q^M$를 도입한다.
- 이전 프레임의 Memory Queries $Q^M_{t-1}$과 현재 프레임의 객체 중심 쿼리 $Q_t$ 간의 내적 유사도를 계산하고, Hungarian Algorithm을 통해 최적의 쌍을 매칭한다.
- 매칭된 쿼리 $Q^*_t$를 사용하여 Memory Queries를 모멘텀 방식으로 업데이트한다:
  $$Q^M_t = \phi^M(Q^M_{t-1}, Q^*_t) = \alpha \cdot s^{obj} \cdot Q^*_t + (1 - \alpha \cdot s^{obj}) \cdot Q^M_{t-1}$$
  여기서 $\alpha$는 업데이트 비율을 조절하는 계수이며, $s^{obj}$는 객체 점수이다. 객체가 가려지거나 사라져 $s^{obj}$가 낮아지면 메모리 업데이트가 억제되어 기존 특징을 유지함으로써 재등장 시 추적을 가능하게 한다.

### 3. Open-Vocabulary Classification
사전 학습된 CLIP Text Encoder $\Phi^{TEXT}$를 사용하여 임의의 카테고리 이름으로부터 텍스트 임베딩 $e_{text}$를 생성한다.
- 추적된 객체의 메모리 쿼리 $Q^M_t$는 Class Head $H_c$를 통해 클래스 임베딩 $e_{cls}$로 변환된다.
- 최종 분류 점수는 $e_{cls}$와 $e_{text}$ 간의 코사인 유사도를 통해 계산된다:
  $$s^{cls}_{i,j} = \sigma(\cos(e^{cls}_i, e^{text}_j)/\epsilon)$$
  여기서 $\sigma$는 시그모이드 함수, $\epsilon$은 온도 하이퍼파라미터이다.

### 훈련 및 손실 함수
OV2Seg는 비디오 데이터셋 대신 LVIS와 같은 이미지 수준 데이터셋에서 훈련되어 효율성을 높였다. 손실 함수는 다음과 같이 구성된다:
$$L_{match}(\hat{y}, y) = \lambda_{obj}L_{obj}(\hat{s}^{obj}, s^{obj}) + \lambda_{cls}L_{cls}(\hat{s}^{cls}, s^{cls}) + \lambda_{mask}L_{mask}(\hat{m}, m)$$
분류 및 객체 점수 손실에는 Binary Cross-Entropy loss를 사용하며, 마스크 손실에는 Dice loss와 Binary Focal loss의 합을 사용한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 제안한 LV-VIS 및 기존의 Youtube-VIS (2019, 2021), OVIS, BURST 데이터셋을 사용하였다.
- **지표**: 전체 카테고리에 대한 mAP와 더불어, 훈련 시 본 베이스 카테고리($mAP_b$)와 보지 못한 노벨 카테고리($mAP_n$)의 성능을 구분하여 측정하였다.
- **비교 대상**: Detic, DetPro 등 오픈 보캐블러리 검출기와 SORT, OWTB 등 추적기를 결합한 '제안-감소-결합' 방식의 베이스라인 모델들과 비교하였다.

### 주요 결과
- **LV-VIS 성능**: ResNet-50 백본 기준, OV2Seg는 LV-VIS 검증 셋에서 $mAP_n$ 11.9를 기록하며, 기존의 Detic-XMem 조합보다 월등한 성능을 보였다.
- **추론 속도**: 클래스 독립적 쿼리를 사용함으로써, 클래스 의존적 쿼리를 사용하는 OV-DETR 대비 추론 속도를 획기적으로 높여 20.1 FPS를 달성하였다.
- **제로샷 일반화**: 타겟 비디오 데이터셋으로의 파인튜닝 없이도 Youtube-VIS2019의 노벨 카테고리에서 $mAP_n$ 11.1을 기록하는 등 강력한 제로샷 일반화 능력을 입증하였다. 특히 OVIS 데이터셋의 심한 가려짐 상황에서도 경쟁력 있는 성능을 보였다.

## 🧠 Insights & Discussion

### 강점
OV2Seg는 Memory Queries를 통해 비디오 전체의 특징을 점진적으로 집계함으로써, 단순한 프레임별 분류 점수 평균 방식보다 견고한 분류 성능을 보인다. 특히 객체가 완전히 가려졌다가 다시 나타나는 상황에서도 추적을 유지하는 능력이 확인되었다.

### 한계 및 비판적 해석
논문에서 언급된 주요 실패 사례는 다음과 같다.
1. **카테고리 충돌(Category Confliction)**: 시각적으로 유사한 노벨 카테고리 객체를 학습 데이터에 포함된 베이스 카테고리로 오분류하는 경향이 있다 (예: 늑대 $\rightarrow$ 개). 이는 오픈 보캐블러리 과제의 근본적인 챌린지이며, 더 넓은 어휘셋의 훈련이나 지식 증류 기법으로 완화 가능할 것으로 보인다.
2. **공통 카테고리의 누락**: '사람(person)'과 같은 매우 흔한 객체의 재현율(recall)이 낮은 경우가 발생한다. 이는 훈련에 사용된 LVIS 데이터셋이 모든 객체를 조밀하게 어노테이션하지 않았기 때문이며, MS-COCO 데이터셋과의 결합을 통해 해결할 수 있을 것으로 판단된다.

## 📌 TL;DR

본 논문은 훈련되지 않은 새로운 카테고리까지 분할, 추적, 분류하는 **Open-Vocabulary Video Instance Segmentation**이라는 새로운 태스크를 정의하고, 이를 위해 대규모 어휘셋을 가진 **LV-VIS 데이터셋**과 end-to-end 모델인 **OV2Seg**를 제안하였다. OV2Seg는 Memory-Induced Transformer를 통해 실시간에 가까운 속도로 강력한 제로샷 일반화 성능을 보여주며, 향후 비디오 분석 및 오픈월드 인지 시스템 연구에 중요한 기여를 할 것으로 기대된다.