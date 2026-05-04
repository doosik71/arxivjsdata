# Instance Segmentation as Image Segmentation Annotation

Thomio Watanabe, Denis Wolf (2019)

## 🧩 Problem to Solve

본 논문은 이미지 내 객체를 정밀하게 검출하고 경계를 획정하는 Instance Segmentation 문제를 해결하고자 한다. 기존의 대부분의 해결책은 Deep Convolutional Neural Networks(CNN)에 의존하고 있으나, 그 구조가 매우 다양하다. 특히 많은 솔루션들이 문제를 여러 개의 하위 작업(subtasks)으로 나누어 처리하는 Multi-task Network 구조나 Proposal-based Network(예: Mask R-CNN) 방식을 채택하고 있다. 이러한 방식은 벤치마크 테스트에서 높은 성능을 보이지만, 수많은 합성곱 층의 사용으로 인해 계산 비용(Computational Cost)이 매우 높다는 단점이 있다.

따라서 본 연구의 목표는 단일 세그멘테이션 네트워크(Single Segmentation Network)만을 사용하여 Instance Segmentation을 수행함으로써 계산 효율성을 높이는 것이다. 이를 위해 객체 정보를 수학적 표현으로 인코딩하는 방식인 DCME(Distance to Center of Mass Encoding) 기법을 확장하여 적용한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Instance Segmentation을 복잡한 네트워크 구조의 문제가 아닌, 일종의 '이미지 세그멘테이션 주석(Annotation)' 문제로 접근하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **단일 네트워크 구조**: Multi-task Network처럼 디코더를 여러 분기로 특수화하지 않고, 단일 세그멘테이션 네트워크만으로 인스턴스 마스크를 생성한다.
2.  **인코더의 재목적화(Repurposing)**: 별도의 분류 네트워크를 두는 대신, 기존 세그멘테이션 네트워크의 인코더를 객체 분류 작업에 함께 활용하여 계산 비용을 획기적으로 줄였다.
3.  **손실 함수 개선**: 큰 해상도의 이미지에서 발생하는 큰 객체의 벡터 값으로 인해 학습이 불안정해지는 문제를 해결하기 위해, 독립적 출력(Independent outputs) 방식과 오차 진폭 클리핑(Error amplitude clipping) 기법을 도입하였다.

## 📎 Related Works

Instance Segmentation 연구는 크게 두 가지 방향으로 나뉜다.

1.  **Proposal-based Networks**: 객체 검출(Object Detection) 네트워크를 통해 바운딩 박스를 먼저 찾고 그 내부에서 마스크를 추출하는 방식이다. Mask R-CNN 및 Path Aggregation Network(PAN)가 대표적이며, 높은 정확도를 보이지만 계산 비용이 매우 크다.
2.  **Proposal-free Networks**: 객체 검출 과정 없이 직접 인스턴스를 분리하는 방식으로, 다양한 인코딩 및 클러스터링 기법을 사용한다.

본 논문은 저자들이 이전에 제안한 DCME 기법을 확장한 것으로, 클래스 구분 없는(Class-agnostic) 인스턴스 마스크를 생성하는 능력을 활용한다. 유사한 인코딩 기법을 제안한 Kendall et al.의 연구가 있으나, 해당 연구는 벡터 맵, 세그멘테이션 맵, 깊이 맵을 동시에 생성하는 Multi-task Network 구조를 사용한다는 점에서 본 논문의 단일 네트워크 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 인코더 재목적화
본 연구는 DeepLabv3+ 네트워크를 기반으로 하며, VGG 인코더와 디콘볼루션(Deconvolution) 층으로 구성된 디코더를 사용한다. 특히 인코더를 단순한 특징 추출기가 아닌, 객체를 대략적으로 위치시키고 분류하는 용도로 재사용한다.

이미지를 격자(Grid) 형태로 나누어 각 격자 블록의 클래스를 추론하며, 이때 격자 크기 $G_s$는 다음과 같이 정의된다.
$$G_s = 2^n$$
이미지 픽셀 위치 $P(x, y)_{image}$와 인코더 출력 위치 $P(x, y)_{encoder}$의 관계는 다음과 같다.
$$P(x, y)_{encoder} = \text{floor}\left(\frac{P(x, y)_{image}}{G_s}\right)$$
최종적으로 인스턴스의 클래스는 DCME를 통해 계산된 인스턴스의 질량 중심(Center of Mass, CM)이 포함된 격자 블록의 클래스를 통해 결정된다.

### 2. 네트워크 디코더 손실 함수 (Loss Function)
DCME는 2D 변위 벡터를 기반으로 하므로, 이미지 해상도가 높아지면 객체의 경계 부분에서 벡터의 크기가 매우 커지는 현상이 발생한다. 이는 학습 시 큰 오차 신호를 생성하여 모델이 큰 객체에만 편향되게 만드는 결과를 초래한다. 이를 해결하기 위해 두 가지 수정 사항을 적용하였다.

**가. 독립적 출력 (Independent Outputs)**
기존의 평균 제곱 오차(MSE) 방식은 모든 출력의 평균값을 사용하여 그래디언트를 업데이트하므로 각 픽셀의 독립적인 특성을 반영하지 못한다.
$$\text{MSE} = \frac{1}{2N} \sum_{i=1}^N (Y_i - \hat{Y}_i)^2$$
본 연구에서는 각 출력 채널의 모든 픽셀을 독립적인 출력으로 간주하여 개별적으로 업데이트한다. 이때 샘플 수 $N$은 다음과 같이 정의된다.
$$N = 2 \cdot n \cdot r \cdot c$$
(여기서 $n$은 이미지 수, $r, c$는 디코더의 공간적 차원, 2는 DCME 출력 채널 수이다.)

**나. 오차 진폭 클리핑 (Error Amplitude Clipping)**
백프로파게이션(Backpropagation) 이전에 오차 값을 클리핑하기 위해 수정된 로지스틱 함수를 사용한다.
$$f(x) = A \cdot \left( \frac{1}{1 + e^{-x}} - 0.5 \right) = \frac{A}{2} \cdot \left( \frac{e^x - 1}{e^x + 1} \right)$$
여기서 $A$는 입력 이미지 해상도에 따라 조정되는 파라미터이며, 이 함수는 원점 주변에서는 선형적으로 동작하고 양 끝단에서는 $\pm A/2$ 값으로 수렴하여 급격한 오차 신호를 억제한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Cityscapes (도시 도로 장면 데이터셋)
- **해상도**: $(512, 1024)$
- **평가 지표**: Average Precision (AP) 및 $AP_{50\%}$
- **비교 대상**: 기존 DCME 방식, Kendall et al.의 Multi-task Learning 방식

### 2. 정량적 결과
- **전체 시스템 성능**: 검증 세트(Validation set)에서 $\text{mean AP} = 11.5$를 달성하였다. 이는 기존 DCME(3.77)보다 크게 향상된 결과이다.
- **테스트 세트 성능**: 테스트 세트에서는 $\text{mean AP} = 7.72$를 기록하였으며, 이는 데이터셋 불균형(Dataset Imbalance)으로 인해 샘플 수가 적은 클래스(truck, bus, train 등)에서 성능 저하가 나타났기 때문이다.
- **타 모델 비교**: Kendall et al.의 방식($\text{mean AP} = 21.57$)보다는 낮지만, 본 모델은 훨씬 단순한 단일 네트워크 구조를 가지므로 계산 효율성이 더 높다.

### 3. 절제 연구 (Ablation Studies)
- **Instance Oracle**: 모든 인스턴스가 완벽하게 검출되었다고 가정하고 인코더의 분류 성능만 평가한 결과 $\text{mean AP} = 25.2$를 기록하였다.
- **Class Oracle**: 모든 클래스 라벨이 정확하다고 가정하고 디코더의 검출 및 경계 획정 성능만 평가한 결과 $\text{mean AP} = 14.0$를 기록하였다.
- **분석**: Class Oracle의 점수가 낮다는 점은 현재 시스템의 병목 지점(Bottleneck)이 분류(Classification)보다는 디코더의 검출 및 묘사(Detection/Delineation) 단계에 있음을 시사한다.

## 🧠 Insights & Discussion

**강점 및 가능성**
본 논문은 Instance Segmentation을 위해 복잡한 Multi-task 구조를 설계하는 대신, 단일 세그멘테이션 네트워크와 효율적인 인코딩(DCME), 그리고 인코더 재사용이라는 전략을 통해 계산 비용을 낮추었다. 특히 부분 가림(Partial Occlusion) 문제에 강건하며, 단순한 구조 덕분에 향후 더 강력한 세그멘테이션 모델이 등장한다면 이에 그대로 적용하여 성능을 높일 수 있는 확장성을 가진다.

**한계 및 비판적 해석**
1. **정밀도 부족**: IoU 임계값이 높아질수록 검출 정확도가 급격히 떨어지는 현상이 관찰되었다. 이는 모델이 객체의 존재는 잘 찾아내지만, 경계를 정밀하게 묘사하는 능력은 부족함을 의미한다.
2. **데이터 불균형**: 테스트 세트 결과에서 나타나듯, 샘플 수가 적은 클래스에 대한 일반화 성능이 매우 낮다. 이는 단순히 네트워크 구조의 문제가 아니라 학습 데이터의 불균형 문제를 해결할 추가적인 기법(예: Data Augmentation 또는 Loss Weighting)이 필요함을 보여준다.
3. **성능 격차**: Proposal-based 방식이나 정교한 Multi-task 방식에 비해 절대적인 AP 수치가 낮다. 효율성 면에서는 이득이 있으나, 고정밀도가 요구되는 실제 자율주행 환경 등에 적용하기에는 추가적인 성능 개선이 필수적이다.

## 📌 TL;DR

본 연구는 복잡한 Multi-task 네트워크 없이 **단일 세그멘테이션 네트워크와 DCME 인코딩**만을 사용하여 Instance Segmentation을 수행하는 효율적인 방법을 제안한다. 특히 **인코더를 분류 작업에 재사용**하고 **손실 함수에 클리핑 기법을 도입**하여 계산 비용을 줄이면서도 학습 안정성을 높였다. 비록 정밀한 경계 묘사 능력과 데이터 불균형 문제는 여전히 과제로 남아있으나, 단순한 구조를 통해 계산 효율성을 극대화했다는 점에서 향후 가벼운 실시간 인스턴스 세그멘테이션 연구에 중요한 기초 아이디어를 제공한다.