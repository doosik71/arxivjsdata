# Vision-Language Model for Object Detection and Segmentation: A Review and Evaluation

Yongchao Feng, et al. (2025)

## 🧩 Problem to Solve

최근 Vision-Language Model(VLM)은 Open-Vocabulary(OV) 객체 탐지 및 세그멘테이션 작업에서 뛰어난 가능성을 보여주었으나, 전통적인 컴퓨터 비전 작업에서의 효과성에 대해서는 여전히 체계적인 평가가 부족한 상태이다. 기존의 리뷰 논문들은 주로 오픈 보캐블러리 설정에만 집중하여, 실제 현실 세계의 복잡한 시나리오에서 VLM이 어떻게 작동하는지에 대한 포괄적인 분석이 이루어지지 않았다.

본 논문의 목표는 VLM을 일종의 '파운데이션 모델(Foundation Model)'로 간주하고, 이를 다양한 다운스트림 비전 작업에 적용하여 그 성능을 체계적으로 평가하는 것이다. 구체적으로는 객체 탐지(Object Detection)와 세그멘테이션(Segmentation)이라는 두 가지 핵심 작업에서 VLM의 한계와 강점을 분석하여, 향후 VLM 설계 및 최적화를 위한 통찰력을 제공하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 기여는 VLM을 단순한 모델이 아닌 범용적인 파운데이션 모델로 정의하고, 이를 다각도로 검증한 광범위한 평가 프레임워크를 구축한 것에 있다.

1. **선구적인 광범위 평가**: 객체 탐지 분야의 8가지 시나리오(폐쇄 집합 탐지, 도메인 적응, 밀집 객체 등)와 세그멘테이션 분야의 8가지 시나리오(Few-shot, 오픈 월드, 소형 객체 등)에 걸쳐 VLM의 성능을 최초로 종합 평가하였다.
2. **미세 조정(Fine-tuning) 전략의 세밀한 분석**: Zero Prediction, Visual Fine-tuning, Text Prompt라는 세 가지 서로 다른 미세 조정 입도(Granularity)가 각 작업의 성능에 미치는 영향을 체계적으로 조사하였다.
3. **내부 메커니즘 분석**: 모델 아키텍처, 학습 방법론, 그리고 작업 특성 간의 상관관계를 분석하여 VLM의 성능을 결정짓는 핵심 요인을 규명하였다.

## 📎 Related Works

논문은 VLM 기반의 탐지 및 세그멘테이션 방법론을 다음과 같이 분류하여 설명한다.

### VLM 기반 객체 탐지 (Object Detection)
- **대규모 사전 학습 기반 방법(Large-scale Pretraining Based Method)**: GLIP, GroundingDINO와 같이 대규모 이미지-텍스트 쌍 데이터를 통해 제로샷 성능을 극대화하는 방식이다.
- **학습 전략 기반 방법(Learning Strategy Based Method)**: 지식 증류(Knowledge Distillation), 의사 라벨 생성(Pseudo-label Generation), 프롬프트 학습(Prompt Learning), LLM 보조 학습 등을 통해 특정 데이터셋의 성능을 높이는 방식이다.

### VLM 기반 세그멘테이션 (Segmentation)
- **완전 지도 학습(Fully-supervised)**: 제한된 카테고리에 대해 정밀한 마스크 어노테이션을 사용하여 학습한다.
- **텍스트 지도 학습(Text-supervised)**: 정밀한 마스크 없이 이미지-텍스트 쌍의 정렬을 통해 영역을 학습한다.
- **학습 불필요 방법(Training-free)**: 사전 학습된 VLM의 특징을 그대로 활용하여 추론 시점에 마스크를 생성한다.

기존 연구들이 주로 Open-Vocabulary 성능 향상에만 매몰되어 있었던 반면, 본 논문은 이를 다양한 실제 제약 조건(노이즈, 도메인 변화, 소형/밀집 객체) 하에서 평가함으로써 차별점을 갖는다.

## 🛠️ Methodology

본 논문은 VLM을 평가하기 위해 세 가지 미세 조정 전략과 다각도의 평가 파이프라인을 설계하였다.

### 미세 조정 전략 (Fine-tuning Granularities)

모델 $f_{\theta}$가 이미지 $x$와 텍스트 프롬프트 $t$를 입력으로 받는다고 할 때, 세 가지 전략은 다음과 같이 정의된다.

1. **Zero Prediction**: 사전 학습된 모델을 수정 없이 그대로 적용한다.
   $$\text{Output} = f_{\theta}(x, t)$$
   이 방식은 계산 비용이 없으며 모델의 고유한 일반화 능력을 평가하는 데 적합하다.

2. **Visual Fine-tuning**: 텍스트 인코더 $E_t$는 고정하고, 시각적 인코더 $E_v$만을 다운스트림 데이터에 맞춰 최적화한다.
   $$\text{Modify } E_v, \text{ keep } E_t \text{ fixed}$$
   시각적 표현력을 직접 최적화하므로 성능 향상 폭이 크지만, 계산 비용이 높다.

3. **Text Prompt**: 시각적 인코더를 고정하고 텍스트 프롬프트에 학습 가능한 파라미터 $\Delta t$를 추가하여 최적화한다.
   $$t' = t + \Delta t$$
   매우 낮은 계산 비용으로 특정 작업에 적응시킬 수 있는 전략이다.

### 평가 프레임워크
- **객체 탐지 평가**: Closed-set, Open-Vocabulary, Fine-grained Perception, Few-shot, Robustness, Domain-Related, Dense Object 등 8개 영역에서 평가한다.
- **세그멘테이션 평가**: Zero-shot, Open-world, Multi-domain(MESS 벤치마크), Fine-grained, Few-shot, Robustness, Dense/Small Object 등 8개 영역에서 평가한다.

## 📊 Results

### 객체 탐지 결과
- **아키텍처의 영향**: Transformer 기반의 DINO/DETR 구조(GroundingDINO, OV-DINO)가 전통적인 Faster R-CNN 구조보다 모든 지표에서 압도적인 성능을 보였다.
- **미세 조정 효율**: Visual Fine-tuning이 Text Prompt보다 특히 복잡한 데이터셋(COCO, LVIS)에서 훨씬 높은 성능을 기록하였다. 이는 시각적 특징 모델링이 성능 향상의 주된 동력임을 시사한다.
- **일반화 능력**: 대규모 사전 학습 모델(OV-DINO 등)이 학습 전략 기반 모델보다 도메인 일반화 성능이 뛰어나며, 이는 사전 학습 데이터의 규모와 다양성이 직접적인 영향을 미침을 보여준다.

### 세그멘테이션 결과
- **지도 방식의 효율**: 제한된 카테고리에 대한 정밀 어노테이션을 사용한 방법(Cat-Seg 등)이 대규모 이미지-텍스트 쌍을 사용한 방법보다 mIoU가 훨씬 높게 나타났다.
- **인식 방식의 차이**: 영역 기반(Region-based) 방법이 픽셀 기반(Pixel-based) 방법보다 전반적으로 우수하지만, 매우 세밀한 부분 세그멘테이션(Fine-grained Part)에서는 픽셀 기반 방식이 더 경쟁력 있는 모습을 보였다.
- **강건성**: CLIP을 고정(Frozen)한 모델이 미세 조정한 모델보다 노이즈 및 변형에 더 강건한(Robust) 특성을 보였다.

## 🧠 Insights & Discussion

### 강점 및 발견점
- **VLM의 파운데이션 모델 가능성**: 대부분의 VLM이 시각적 미세 조정을 통해 다양한 다운스트림 작업에서 높은 성능을 달성함으로써, VLM이 범용적인 시각 인식 파운데이션 모델로 기능할 수 있음을 입증하였다.
- **아키텍처의 중요성**: 모델의 규모보다 하위 디텍터 아키텍처(예: DINO vs Faster R-CNN)가 최종 성능에 더 결정적인 영향을 미친다는 점을 확인하였다.

### 한계 및 비판적 해석
- **세밀한 인식의 한계**: 제로샷 설정에서 VLM은 미세 분류(Fine-grained) 작업(예: 강아지 품종 구분)에서 매우 저조한 성능을 보였다. 이는 사전 학습 단계의 세밀함(Granularity)이 추론 단계의 요구사항과 일치하지 않기 때문이며, VLM이 아직 계층적 다중 입도 세만틱 이해 능력을 갖추지 못했음을 의미한다.
- **치명적 망각(Catastrophic Forgetting)**: 베이스 카테고리에 대해 시각적 미세 조정을 수행할 경우, 오히려 새로운(Novel) 카테고리에 대한 일반화 성능이 떨어지는 현상이 관찰되었다.

### 향후 연구 방향
- **사전 학습 패러다임 최적화**: 다운스트림 작업의 특성(공간적 관계 모델링 등)을 사전 학습 단계에서 명시적으로 고려하는 설계가 필요하다.
- **효율적인 특징 융합**: 시각-텍스트 특징의 단순 정렬을 넘어, 백본 단계에서부터 조기에 융합하는(Early Fusion) 경량화된 구조 설계가 요구된다.

## 📌 TL;DR

본 논문은 VLM을 파운데이션 모델로 정의하고 객체 탐지와 세그멘테이션의 16가지 시나리오에서 체계적으로 평가하였다. 분석 결과, Transformer 기반 아키텍처의 중요성과 Visual Fine-tuning의 강력한 효과를 확인하였으며, 특히 제로샷 기반의 미세 인식(Fine-grained) 능력 부족이라는 핵심 한계를 규명하였다. 이 연구는 VLM을 실제 산업 현장의 복잡한 비전 작업에 적용하기 위해 어떤 아키텍처와 튜닝 전략을 선택해야 하는지에 대한 정량적인 가이드라인을 제공한다.