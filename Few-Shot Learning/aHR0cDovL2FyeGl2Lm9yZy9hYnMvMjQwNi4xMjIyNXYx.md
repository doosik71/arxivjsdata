# The Solution for CVPR2024 Foundational Few-Shot Object Detection Challenge

Hongpeng Pan, Shifeng Yi, Shouwei Yang, Lei Qi, Bing Hu, Yi Xu, Yang Yang (2024)

## 🧩 Problem to Solve

본 논문은 Foundational Few-Shot Object Detection (FSOD) 작업에서 Vision-Language Model (VLM)이 겪는 **개념 불일치(Concept Misalignment)** 문제를 해결하고자 한다.

최근 GLIP나 Grounding DINO와 같은 VLM은 텍스트 프롬프트를 통해 Zero-shot 객체 검출이 가능하다는 강력한 장점이 있다. 그러나 자율 주행 인식과 같은 특정 도메인 데이터셋에서는 VLM이 인식하는 대상과 실제 정답(Target Concept) 사이의 불일치가 발생한다. 예를 들어, 'barrier'라는 클래스를 인식할 때 VLM은 도로의 바리케이드뿐만 아니라 도로변의 계단까지 함께 검출하는 식의 오작동이 발생할 수 있다.

이러한 불일치는 VLM의 Zero-shot 성능을 저하시킬 뿐만 아니라, 의사 라벨(Pseudo-labels)을 생성하여 모델을 미세 조정(Fine-tuning)하는 방식의 접근법에서 잘못된 라벨을 생성하게 만들어 전체적인 성능 하락으로 이어진다. 따라서 본 연구의 목표는 시각적 개념과 텍스트 개념 사이의 간극을 줄여 정밀한 의사 라벨을 생성하고, 이를 통해 Few-shot 환경에서의 검출 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **MM-LLM(Multimodal Large Language Model)을 활용하여 클래스 이름보다 더 구체적인 '참조 표현(Referential Expressions)'을 생성하고, 이를 통해 VLM의 개념 이해도를 높이는 VLM+ 프레임워크**를 제안한 것이다.

단순히 클래스 명칭(예: 'debris')을 사용하는 대신, GPT-4와 같은 MM-LLM이 해당 객체의 시각적 특성을 묘사한 여러 표현을 생성하게 한다. 이후 실제 정답 데이터(Ground Truth)와의 IoU(Intersection over Union)를 측정하여 해당 클래스를 가장 정확하게 검출해내는 최적의 참조 표현을 선택한다. 이렇게 최적화된 표현을 기반으로 고품질의 의사 라벨을 생성하고, 이를 기존 라벨 데이터와 결합하여 모델을 반복적으로 학습시키는 전략을 취한다.

## 📎 Related Works

논문에서는 Open-set 객체 검출의 대표적인 모델로 **GLIP**와 **Grounding DINO**를 언급한다.

- **GLIP**: 객체 검출 작업을 문맥이 없는 구절 지역화(Context-free phrase localization) 작업으로 재정의하여, 언어적 일반화를 통해 임의의 클래스를 검출할 수 있게 한다.
- **Grounding DINO**: Transformer 기반의 DINO 검출기와 Grounded pre-training을 결합한 모델이다. 텍스트 가이드 쿼리 선택 및 교차 모달 디코더를 통해 텍스트와 시각적 정보를 효과적으로 융합한다.

기존 접근 방식과의 차별점은, 기존 VLM들이 제공된 텍스트 프롬프트를 그대로 수용하여 Zero-shot 검출을 수행하는 반면, VLM+는 MM-LLM을 이용해 특정 데이터셋의 도메인에 최적화된 텍스트 표현을 먼저 찾고, 이를 통해 생성된 정제된 의사 라벨로 모델을 최적화한다는 점에 있다.

## 🛠️ Methodology

### 1. 전체 파이프라인
VLM+의 전체 프로세스는 **[개념 정렬(Concept Alignment) $\rightarrow$ 의사 라벨 생성 $\rightarrow$ 반복적 모델 최적화]** 순으로 진행된다.

### 2. 개념 정렬 (Concept Alignment)
VLM이 클래스 명칭의 모호함으로 인해 대상을 잘못 인식하는 문제를 해결하기 위해 다음 절차를 수행한다.

1. **참조 표현 생성**: 학습 세트의 이미지에 정답 바운딩 박스를 표시한 후, GPT-4에 "빨간 박스 안의 객체에 대한 5가지 묘사 용어를 제공하라"고 요청하여 각 카테고리별 묘사 프롬프트를 얻는다.
2. **최적 표현 선택**: 생성된 프롬프트들을 무작위로 조합하여 $N$개의 참조 표현 $T^c_i$를 만든다. 각 표현을 VLM에 입력하여 얻은 예측 박스 $P^c_{i,j}$와 실제 정답 박스 $B^c_j$ 사이의 IoU를 계산한다.
   $$IoU(P^c_{i,j}, B^c_j) = \frac{|P^c_{i,j} \cap B^c_j|}{|P^c_{i,j} \cup B^c_j|}$$
3. **정확도 측정**: IoU가 0.5보다 큰 경우를 성공으로 간주하는 지시 함수를 사용하여 정확도 $ACC$를 계산한다.
   $$ACC(P^c_{i,j}, B^c_j) = \frac{1}{10} \sum_{j=1}^{10} \mathbb{1}_{IoU(P^c_{i,j}, B^c_j) > 0.5}$$
4. **최종 선택**: 가장 높은 정확도를 기록한 참조 표현 $T^{c*}_{i}$를 해당 클래스의 대표 표현으로 선택한다.

### 3. 반복적 의사 라벨 최적화 (Iterative Pseudo-label Optimization)
최적화된 참조 표현을 사용하여 모델의 성능을 점진적으로 향상시킨다.

1. **초기 생성**: 최적의 참조 표현을 사용하여 레이블이 없는 데이터에 대해 초기 의사 라벨을 생성한다. 이때 신뢰도 점수가 임계값 $\eta$ (본 논문에서는 0.3)를 초과하는 경우에만 라벨로 인정한다.
2. **모델 학습**: 정답 라벨 데이터와 생성된 의사 라벨 데이터를 함께 사용하여 VLM을 학습시킨다.
3. **라벨 정제**: 학습된 모델을 사용하여 다시 의사 라벨을 생성한다. 모델이 업데이트되었으므로 이전보다 더 정확한 라벨이 생성된다.
4. **반복**: 수렴 조건에 도달하거나 정해진 반복 횟수까지 학습과 정제 과정을 반복한다.

### 4. 손실 함수 (Loss Function)
학습 시에는 다음과 같은 손실 함수들의 가중치 합을 사용한다.
- **Focal Loss**: 클래스 불균형 문제를 해결하기 위해 사용하며, 가중치는 1.0이다.
- **Box L1 Loss**: 바운딩 박스의 좌표 오차를 줄이기 위해 사용하며, 가중치는 5.0이다.
- **GIOU Loss**: 박스 간의 겹침 정도를 최적화하며, 가중치는 2.0이다.
(Grounding DINO의 경우, DETR 구조와 유사하게 각 디코더 층과 인코더 출력 이후에 보조 손실(Auxiliary losses)을 추가로 적용한다.)

## 📊 Results

### 실험 설정
- **데이터셋**: CVPR2024 Foundational Few-Shot Object Detection Challenge 제공 데이터.
- **비교 모델**: GLIP, Grounding DINO (Zero-shot 및 VLM+ 적용 버전).
- **평가 지표**: mAP (mean Average Precision).

### 정량적 결과
실험 결과, VLM+ 프레임워크를 적용했을 때 모든 모델에서 성능 향상이 나타났다. 특히 Grounding DINO에서 그 효과가 두드러졌다.

| Methods | mAP |
| :--- | :---: |
| Baseline (Best) | 21.51 |
| GLIP (zero-shot) | 15.73 |
| **GLIP +** | **27.27** |
| Grounding DINO (zero-shot) | 19.91 |
| **Grounding DINO +** | **32.56** |

### 분석 및 사례 연구
- **개념 정렬 효과**: Table 1에 따르면 'debris' 클래스의 경우, 단순 클래스 명칭을 사용했을 때의 정확도는 0이었으나, "indicator warning board with wooden frame"이라는 참조 표현을 사용했을 때 정확도가 0.7까지 상승하였다.
- **시각적 확인**: Figure 4의 사례 연구에서 'personal mobility'나 'pushable pullable' 같은 모호한 용어 대신 'small kick scooter'나 'pushable pullable garbage container'와 같은 구체적인 표현을 사용했을 때 VLM이 훨씬 더 정확하게 객체를 검출함을 확인하였다.

## 🧠 Insights & Discussion

본 논문의 강점은 VLM의 텍스트 입력부에 대한 **'데이터 중심적 최적화(Data-centric optimization)'**를 수행했다는 점이다. 모델 아키텍처를 수정하는 대신, MM-LLM을 활용해 VLM이 가장 잘 이해할 수 있는 형태의 텍스트를 찾아내어 입력함으로써 Zero-shot의 한계를 극복하였다.

다만, 몇 가지 한계점과 논의 사항이 존재한다.
첫째, 최적의 참조 표현을 찾기 위해 실제 정답(Ground Truth) 데이터의 일부(10개 샘플)를 사용한다. 이는 엄격한 Zero-shot 설정과는 거리감이 있으며, 소량의 감독 학습(Supervised) 성격이 포함된 방식이다.
둘째, 참조 표현을 생성하는 과정에서 GPT-4와 같은 외부 MM-LLM에 의존하므로, API 비용이나 외부 모델의 성능에 영향을 받는다.
셋째, 반복적 의사 라벨링 과정에서 발생할 수 있는 'Confirmation Bias(모델이 자신의 틀린 예측을 정답으로 믿고 계속 학습하는 현상)'에 대한 명시적인 방지책이 논문에 충분히 서술되지 않았다.

## 📌 TL;DR

본 연구는 VLM의 클래스 명칭 모호성으로 인한 검출 성능 저하 문제를 해결하기 위해, **MM-LLM(GPT-4)으로 구체적인 참조 표현을 생성하고 IoU 기반으로 최적의 표현을 선택하여 학습에 활용하는 VLM+ 프레임워크**를 제안하였다. 이를 통해 Grounding DINO 기준 mAP를 19.91에서 32.56으로 크게 향상시켰으며, 이는 특정 도메인의 Few-shot 객체 검출 작업에서 텍스트 프롬프트의 정교한 설계와 반복적인 의사 라벨 정제가 매우 중요하다는 것을 시사한다.