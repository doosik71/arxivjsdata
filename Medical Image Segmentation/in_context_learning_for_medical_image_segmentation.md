# IN-CONTEXT LEARNING FOR MEDICAL IMAGE SEGMENTATION

Eichi Takaya and Shinnosuke Yamamoto (2025)

## 🧩 Problem to Solve

의료 영상(MRI, CT 등)의 정확한 어노테이션(Annotation)은 치료 효과 평가 및 방사선 치료 계획 수립에 필수적이다. 그러나 이러한 작업은 의료 전문가의 상당한 시간과 노력을 요구하며, 이로 인해 학습에 사용할 수 있는 레이블링된 데이터의 양이 제한되어 의료 영상 분야의 AI 적용에 큰 제약이 되고 있다.

특히 CT나 MRI와 같은 연속적인 슬라이스(Sequential slices) 데이터의 경우, 각 슬라이스를 독립적으로 분할(Segmentation)하면 인접한 슬라이스 간의 공간적 일관성(Spatial consistency)이 유지되지 않는 문제가 발생한다. 본 논문의 목표는 최소한의 어노테이션만으로도 높은 분할 정확도를 유지하면서, 연속적인 의료 영상 간의 일관성을 보장하는 새로운 분할 방법론을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **In-context Cascade Segmentation (ICS)**라는 새로운 전략을 제안한 것이다.

기존의 Few-shot segmentation 모델인 UniverSeg를 기반으로 하되, 추론 단계에서 각 슬라이스의 예측 결과(Predicted masks)를 다시 Support set(참조 데이터셋)에 동적으로 포함시키는 '캐스케이드(Cascade)' 방식을 도입하였다. 이를 통해 정보를 앞뒤 방향으로 전파함으로써 슬라이스 간의 일관성을 확보하고, 추가적인 모델 재학습(Re-training) 없이도 분할 성능을 향상시킨 것이 핵심적인 직관이다.

## 📎 Related Works

### 1. 의료 영상 분할 (Medical Image Segmentation)

초기에는 K-means나 Watershed 같은 비지도 학습 기반의 전통적인 머신러닝 방법이 사용되었으나, 정확도와 확장성에 한계가 있었다. 이후 U-Net과 같은 Encoder-Decoder 구조의 딥러닝 모델이 표준이 되었으며, 최근에는 Vision Transformer(ViT) 기반 모델과 MedSAM과 같은 거대 파운데이션 모델(Foundation Models)이 등장하여 적은 양의 데이터로도 높은 성능을 보이는 추세이다.

### 2. In-context Learning (ICL)

원래 LLM에서 제안된 개념으로, 추가 학습 없이 몇 개의 예시(Prompt)만으로 새로운 태스크에 적응하는 능력이다. UniverSeg는 이를 의료 영상 분할에 적용하여, 레이블링된 'Support images'를 통해 쿼리 이미지의 분할을 가이드하는 방식을 취한다.

### 3. Semi-supervised Learning 및 4S

레이블이 부족한 상황에서 미레이블 데이터를 활용하는 방법이다. 특히 **Sequential semi-supervised segmentation (4S)**는 소수의 연속된 레이블 슬라이스로 시작해 추론된 마스크를 의사 레이블(Pseudo-labels)로 사용하여 영역을 확장하는 방식과 유사하다. 하지만 4S는 반복적인 모델 재학습이 필요하여 계산 비용이 매우 높은 반면, 제안된 ICS는 In-context learning을 활용하여 파라미터 업데이트 없이 Support set만 갱신하므로 계산 효율성이 훨씬 높다.

## 🛠️ Methodology

### 1. 문제 정의 (Problem Definition)

전체 볼륨 데이터셋 $V = \{\text{slice}_1, \text{slice}_2, \dots, \text{slice}_n\}$이 주어졌을 때, 이 중 $m$개의 슬라이스만 레이블이 존재한다고 가정한다. 이 레이블된 집합을 Support set $S$라고 정의한다.
$$S = \{(\text{slice}_{\ell_1}, \text{label}_{\ell_1}), (\text{slice}_{\ell_2}, \text{label}_{\ell_2}), \dots, (\text{slice}_{\ell_m}, \text{label}_{\ell_m})\}$$
목표는 모델 $f$를 이용하여 레이블이 없는 나머지 $(n-m)$개 슬라이스에 대해 예측 마스크 $\hat{M}_k$를 생성하는 것이다.
$$\hat{M}_k = f(\text{slice}_k, S)$$

### 2. UniverSeg 프레임워크

본 연구의 기반이 되는 UniverSeg는 거대 의료 영상 데이터셋(MegaMedical)으로 사전 학습된 모델이다. Encoder-Decoder 구조를 가지며, 쿼리 이미지와 Support 이미지 간의 양방향 특징 교환을 위해 **CrossBlock** 모듈을 사용한다. 추가 학습 없이 Support set만으로 Few-shot 분할을 수행한다.

### 3. In-context Cascade Segmentation (ICS)

ICS는 UniverSeg의 독립적인 추론 방식에서 발생하는 불일치 문제를 해결하기 위해 다음과 같은 절차를 따른다.

1. **Support Set 초기화**: 볼륨의 중앙부 또는 해부학적으로 중요한 영역에서 소수의 레이블된 슬라이스를 선택하여 초기 Support set을 구성한다.
2. **양방향 순차 추론 (Bidirectional Sequential Inference)**:
    * **Forward Pass**: 초기 세트부터 마지막 슬라이스까지 순방향으로 추론을 진행한다. 이때, 방금 추론한 $\hat{M}_k$를 다음 슬라이스 추론을 위한 Support set에 즉시 추가한다.
    * **Backward Pass**: 동일한 방식으로 마지막 슬라이스부터 첫 번째 슬라이스까지 역방향으로 추론을 진행한다.
3. **Support Set 크기 제한**: Support set이 무한히 커지는 것을 방지하기 위해, 최신 $m$개의 슬라이스만 유지하고 오래된 데이터는 삭제하는 슬라이딩 윈도우 방식을 사용한다.
4. **데이터 증강**: Support set의 강건성을 높이기 위해 $90^\circ, 180^\circ, 270^\circ$ 회전 증강을 적용한다.

## 📊 Results

### 1. 실험 설정

* **데이터셋**: HVSMR (심혈관 MRI 스캔 60건). 8개 해부학적 영역(LV, RV, LA, RA, AO, PA, SVC, IVC)을 대상으로 함.
* **평가 지표**: Dice Similarity Coefficient (DSC)를 사용한다.
$$\text{DSC} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
* **비교 대상**: Baseline(독립적으로 추론하는 표준 UniverSeg) vs ICS.

### 2. 주요 결과

* **정량적 성과**: LA, RA, AO, PA, SVC 영역에서 ICS가 Baseline보다 유의미하게 높은 DSC를 기록하였다 ($p < 0.05$). 반면 LV, RV, IVC 영역에서는 큰 차이가 없었다.
* **정성적 성과**: PA(폐동맥)와 같이 구조가 복잡하고 슬라이스 간 위치 변화가 심한 영역에서 ICS는 끊김 없는 일관된 마스크를 생성한 반면, Baseline은 예측이 불연속적이거나 누락되는 부분이 많았다.
* **초기 Support set의 영향**:
  * **개수 ($m$)**: $m$이 1에서 5로 증가함에 따라 전반적인 DSC가 상승하는 경향을 보였다.
  * **위치**: 초기 레이블 슬라이스의 위치에 따라 정확도 편차가 크게 나타났다. 특히 초기 위치에서 멀어질수록 Ground Truth와의 오차가 커지는 경향이 확인되었다.

## 🧠 Insights & Discussion

### 1. 강점 및 한계

* **강점**: 추가적인 파라미터 학습 없이도 추론 단계의 전략만으로 슬라이스 간 일관성을 획기적으로 개선하였다. 특히 해부학적으로 복잡한 구조의 연속성을 유지하는 데 탁월하다.
* **한계**: 일부 영역(LV 등)에서 **과분할(Over-segmentation)** 경향이 나타나 False Positive가 증가하는 현상이 발견되었다. 이는 Pseudo-label을 사용하는 Self-training 방식의 전형적인 문제와 유사하며, 향후 신뢰도 점수(Confidence metrics) 도입이 필요해 보인다.

### 2. 비판적 해석 및 논의

* **계산 효율성 vs 성능**: $m$을 늘리면 성능이 향상되지만, GPU 메모리 사용량과 추론 시간이 증가하는 트레이드-오프가 존재한다. 따라서 효율적인 모델 증류(Distillation)나 최적의 $m$ 값 탐색이 필요하다.
* **Cold-start 문제**: 초기 Support slice를 어디서 선택하느냐가 결과에 지대한 영향을 미친다. 현재는 수동 또는 임의 선택에 의존하고 있으나, Active Learning의 Cold-start 문제처럼 클러스터링 기반의 자동 선택 기법이 도입되어야 실용성이 높아질 것이다.
* **일반화 검증**: HVSMR 데이터셋 하나만 사용했다는 점이 한계이며, CT나 초음파 등 다른 모달리티에서도 동일한 효과가 있는지 검증이 필요하다.

## 📌 TL;DR

본 논문은 의료 영상의 레이블 부족 문제를 해결하기 위해, UniverSeg 모델을 기반으로 추론 결과를 실시간으로 참조 데이터에 추가하는 **In-context Cascade Segmentation (ICS)** 방법을 제안하였다. 이 방법은 양방향 순차 추론을 통해 슬라이스 간의 해부학적 일관성을 유지하며, 특히 복잡한 혈관 구조 분할에서 성능 향상을 입증하였다. 이는 의료 영상 전문가의 어노테이션 부담을 획기적으로 줄이면서도 강건한 분할 결과를 얻을 수 있는 유망한 접근법이며, 향후 최적의 초기 슬라이스 선택 알고리즘과 결합될 때 임상 적용 가능성이 높을 것으로 기대된다.
