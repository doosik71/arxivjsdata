# Vision Mamba: Cutting-Edge Classification of Alzheimer’s Disease with 3D MRI Scans

Muthukumar K A, Amit Gurung, Priya Ranjan (2024)

## 🧩 Problem to Solve

본 논문은 3D MRI 스캔 데이터를 활용하여 알츠하이머병(Alzheimer's Disease, AD)을 조기에 진단하고 분류하는 문제를 해결하고자 한다. 3D MRI 데이터는 뇌 구조의 미세한 변화를 포착할 수 있는 중요한 도구이지만, 데이터의 고차원성과 복잡성으로 인해 기존의 딥러닝 모델을 적용하는 데 다음과 같은 한계가 존재한다.

첫째, 합성곱 신경망(Convolutional Neural Networks, CNNs)은 지역적인 공간 특징(local spatial features) 추출에는 효과적이지만, 뇌 구조의 전체적인 맥락을 이해하는 데 필요한 장거리 의존성(long-range dependencies)을 캡처하는 능력이 부족하며, 고해상도 3D 데이터를 처리할 때 막대한 계산 자원이 소모된다.

둘째, 트랜스포머(Transformers) 모델은 전역적 문맥(global context) 파악에 뛰어나지만, 셀프 어텐션(self-attention) 메커니즘의 계산 복잡도가 입력 크기에 대해 제곱 비례(quadratic complexity)하여 메모리 사용량이 급증하고 추론 시간이 길어진다는 단점이 있다.

따라서 본 연구의 목표는 이러한 계산 효율성과 전역적 특징 추출 능력 사이의 트레이드오프를 해결하기 위해 State Space Models(SSMs) 기반의 Vision Mamba 아키텍처를 도입하여, 3D MRI 데이터 기반의 알츠하이머병 분류 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 State Space Models(SSMs)의 효율적인 시퀀스 모델링 능력을 3D 의료 영상 분석에 적용하는 것이다. 주요 기여 사항은 다음과 같다.

1. **SSM 기반의 3D MRI 분류 모델 제안**: 고차원 의료 영상 데이터의 특성에 맞게 설계된 Vision Mamba 아키텍처를 통해 계산 효율성을 높이면서도 전역적인 공간 정보를 효과적으로 유지한다.
2. **하이브리드 처리 구조**: 학습 단계에서는 합성곱 연산의 병렬 가능성(parallelizable nature)을 활용하여 속도를 높이고, 추론 단계에서는 순환적 상태 처리(recurrent processing)를 통해 효율적으로 추론을 수행하는 구조를 채택하였다.
3. **동적 상태 표현 및 선택적 스캔**: 입력 특징에 따라 상태 전이를 동적으로 조정하는 Selective Scan 알고리즘을 통해 3D 볼륨 내에서 중요한 공간 정보만을 선택적으로 유지함으로써 진단 정확도를 높였다.

## 📎 Related Works

기존의 알츠하이머병 진단 연구는 주로 CNN과 Transformer에 의존해 왔다.

* **CNN 기반 접근**: Residual CNN 등을 사용하여 3D MRI에서 알츠하이머 단계별 분류를 시도하였으며, 지역적 특징 추출 능력을 통해 성과를 거두었다. 그러나 앞서 언급한 바와 같이 고해상도 3D 데이터 처리 시의 효율성 저하와 전역적 문맥 파악의 한계가 명확하다.
* **Transformer 기반 접근**: Vision Transformer(ViT)나 Swin Transformer와 같이 계층적 구조를 도입하여 CNN의 한계를 극복하고 장거리 의존성을 캡처하려 하였다. 하지만 3D 데이터의 크기가 커질수록 어텐션 연산 비용이 기하급수적으로 증가하여 실제 임상 적용에 제약이 있다.
* **차별점**: 본 논문은 SSM을 도입함으로써 CNN의 병렬 학습 능력과 RNN의 효율적인 추론 능력을 동시에 확보한다. 특히 $\mathcal{O}(N)$의 선형 복잡도로 장거리 의존성을 처리할 수 있어, 기존 모델들보다 대규모 3D MRI 데이터를 훨씬 효율적으로 처리할 수 있다.

## 🛠️ Methodology

### 전체 시스템 구조

Vision Mamba는 입력 3D MRI 영상을 패치 단위로 분할하여 임베딩한 후, 이를 SS-Conv-SSM 블록과 패치 병합(Patch Merging) 층을 거쳐 최종 분류하는 파이프라인을 가진다.

1. **Patch Embedding**: $224 \times 224 \times 160$ 크기의 3D MRI 영상을 겹치지 않는 패치로 나누고, 선형 투영을 통해 고차원 공간으로 매핑한다.
    $$Patch Embedding(x) = x' \in \mathbb{R}^{N \times C}$$
    여기서 $N$은 패치의 개수, $C$는 임베딩 차원이다.

2. **SS-Conv-SSM Block**: 이 블록은 두 개의 병렬 브랜치로 구성된다.
    * **Conv-Branch**: Batch Normalization $\rightarrow$ 3D Convolution $\rightarrow$ ReLU 순으로 연산하여 지역적 특징을 추출한다.
        $$\text{Conv-Branch}(x') = \text{Conv3D}(\text{ReLU}(\text{BN}(x')))$$
    * **SSM-Branch**: Layer Normalization $\rightarrow$ Linear $\rightarrow$ SiLU $\rightarrow$ 3D Selective Scan(SS3D) 순으로 연산하여 전역적 의존성을 캡처한다.
        $$\text{SSM-Branch}(x') = \text{SS3D}(\text{SiLU}(\text{Linear}(\text{LN}(x'))))$$
    두 브랜치의 출력은 채널 차원을 따라 병합(merge)된다.

3. **Patch Merging**: 공간적 차원을 줄이고 채널 수를 늘려 정보를 압축하며 계산 복잡도를 낮춘다.

4. **Classification**: 최종적으로 Fully Connected(FC) 층과 Softmax 함수를 통해 클래스별 확률을 출력한다.
    $$y = \text{Softmax}(\text{FC}(x'))$$

### 핵심 알고리즘 및 방정식

**1. Dynamic State Space Model (SSM)**
상태 전이는 다음과 같은 상태 방정식에 의해 정의된다.
$$h_{t+1} = Ah_t + Bx_t$$
$$y_t = Ch_t + Dx_t$$
여기서 $h_t$는 시간 $t$에서의 상태, $A$는 상태 전이 행렬, $B$는 입력 행렬, $x_t$는 입력, $C$는 출력 행렬, $D$는 직접 매핑 행렬이다. Vision Mamba는 $B$와 $C$ 행렬을 입력 특징에 따라 변화하는 **동적 행렬(Dynamic Matrices)**로 설정하여 중요 정보에 집중하게 한다.

**2. Selective Scan Algorithm**
이 알고리즘은 입력 데이터의 중요도에 따라 상태 전이를 동적으로 조정하여 불필요한 정보는 필터링하고 핵심적인 공간 특징만을 보존한다.

**3. HiPPO Initialization**
장거리 의존성 관리를 위해 High-order Polynomial Projection Operators(HiPPO) 초기화를 사용한다. 이를 통해 상태 전이 행렬 $A$가 최신 정보를 우선시하면서 오래된 정보를 점진적으로 감쇠시켜 3D MRI의 복잡한 구조를 정확하게 분석할 수 있게 한다.

**4. 학습 목표 및 손실 함수**
모델은 예측값 $\hat{y}$와 실제 라벨 $y$ 사이의 차이를 최소화하기 위해 Cross-Entropy(CE) 손실 함수를 사용한다.
$$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{CE}(y_i, \hat{y}_i)$$

## 📊 Results

### 실험 설정

* **데이터셋**: ADNI(Alzheimer’s Disease Neuroimaging Initiative) 데이터셋을 사용하였다.
* **데이터 구성**: 총 3,020개의 3D MRI 스캔 데이터 ($224 \times 224 \times 160$ voxels)를 활용하였으며, 클래스는 AD(알츠하이머병), MCI(경도인지장애), CN(정상 인지)의 세 가지로 구분된다.
* **데이터 분할**:
  * Training Set A: 1,833개 (80%)
  * Test Set A: 461개 (20%)
  * Test Set B: 306개 (외부 검증용)
  * Test Set C: 420개 (외부 검증용)
* **평가 지표**: Accuracy, Precision, Recall, F1 Score를 측정하였다.

### 주요 결과

Vision Mamba의 데이터셋별 정확도는 다음과 같다.

* **Dataset A**: 65%
* **Dataset B**: 44%
* **Dataset C**: 48%

특히 Dataset A에서 AD(Precision 0.73, Recall 0.40, F1 0.51), CN(Precision 0.74, Recall 0.48, F1 0.58), MCI(Precision 0.62, Recall 0.87, F1 0.72)의 성과를 보였다.

### 모델 비교 결과 (Accuracy)

| 모델 | Dataset A | Dataset B | Dataset C |
| :--- | :---: | :---: | :---: |
| **Vision Mamba** | **0.65** | **0.44** | **0.48** |
| Vision Transformer | 0.46 | 0.43 | 0.49 |
| CNN | 0.49 | 0.43 | 0.49 |

Vision Mamba는 특히 Dataset A에서 기존 CNN 및 Vision Transformer보다 유의미하게 높은 정확도를 기록하였으며, 전반적으로 우수하거나 대등한 성능을 보였다.

## 🧠 Insights & Discussion

**강점 및 성과**
본 연구에서 Vision Mamba는 특히 **MCI(경도인지장애)의 높은 Recall(재현율)**을 보였다. 이는 알츠하이머병으로 발전하기 전 단계인 초기 징후를 포착하는 능력이 뛰어나다는 것을 의미하며, 조기 진단 및 치료 계획 수립에 있어 매우 중요한 가치를 가진다.

**한계 및 비판적 해석**

1. **AD 분류 성능 저하**: MCI에 비해 AD(알츠하이머병) 단계의 Precision과 Recall이 낮게 나타났다. 이는 모델이 초기 징후는 잘 잡지만, 이미 진행된 알츠하이머의 복잡한 패턴을 구분하는 데에는 아직 어려움이 있음을 시사한다.
2. **일반화 성능 문제**: Training set A에서는 65%의 정확도를 보였으나, 외부 데이터셋인 B, C에서는 40%대로 급격히 하락하였다. 이는 모델이 학습 데이터에 과적합(overfitting)되었을 가능성이 있으며, 데이터 증강(Data Augmentation) 등의 추가적인 기법이 필요함을 보여준다.
3. **계산 복잡도**: 3D 데이터의 특성상 여전히 높은 계산 복잡도를 가지므로, 실시간 적용을 위해서는 하드웨어 가속 및 추가적인 최적화가 필수적이다.

**향후 발전 방향**
저자들은 MRI 외에도 PET 스캔이나 뇌척수액(CSF) 바이오마커와 같은 **멀티모달(multi-modal) 데이터**를 통합한다면 진단 정확도를 더욱 높일 수 있을 것이라고 제안한다.

## 📌 TL;DR

본 논문은 3D MRI 기반의 알츠하이머병 진단을 위해 **State Space Model(SSM) 기반의 Vision Mamba** 아키텍처를 제안하였다. 이 모델은 CNN의 지역적 특징 추출 능력과 SSM의 효율적인 전역적 의존성 캡처 능력을 결합하여, 기존 CNN 및 ViT보다 우수한 분류 성능을 보였다. 특히 **조기 진단의 핵심인 MCI(경도인지장애) 탐지에서 높은 재현율**을 기록하여 임상적 유용성을 입증하였으며, 향후 고차원 의료 영상 분석의 효율적인 대안이 될 가능성이 크다.
