# Deep Siamese Networks with Bayesian non-Parametrics for Video Object Tracking

Anthony D. Rhodes, Manan Goel (2018)

## 🧩 Problem to Solve

비디오 객체 추적(Video Object Tracking)의 핵심은 첫 번째 프레임에서 지정된 바운딩 박스(bounding-box) 내의 객체를 이후 프레임에서도 정확하게 찾아내는 것이다. 이를 위해서는 강력한 유사도 함수(similarity function)와 이후 프레임에서 객체가 존재할 가능성이 높은 위치를 효율적으로 탐색하는 쿼리 방법이 동시에 필요하다.

기존의 딥러닝 기반 추적 모델들은 두 가지 주요 문제점을 가지고 있다. 첫째, 많은 모델이 온라인 학습(online training)을 필요로 하여 실제 적용 시 처리 속도가 너무 느리다. 둘째, 오프라인으로 학습된 모델들 중 상당수는 분류(classification) 기반 접근 방식을 취하고 있어 특정 클래스에 국한된 탐색만 가능하거나, 객체 위치를 찾기 위해 너무 많은 이미지 패치를 네트워크에 통과시켜야 하는 비효율성이 존재한다.

본 논문의 목표는 딥러닝의 강력한 표현 학습 능력과 고전적인 통계적 최적화 기법을 결합하여, 온라인 학습 없이도 일반화 성능이 높고 탐색 효율성이 뛰어난 객체 추적 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 딥러닝 기반의 **Siamese Network**를 일반적인 객체 유사도 함수로 사용하고, 이를 **Bayesian Optimization (BO)** 프레임워크와 결합하여 시공간적(spatio-temporal) 정보를 인코딩하는 것이다.

특히, 비디오 추적 문제를 시간에 따라 변화하는 **Dynamic Optimization Problem (DOP)**으로 정의하고, **Gaussian Process (GP)**를 사용하여 각 프레임에서 객체의 위치를 나타내는 목적 함수(objective function)를 모델링한다. 이를 통해 탐색 공간을 통계적으로 원칙에 맞게(principled) 그리고 효율적으로 쿼리함으로써, 불필요한 연산을 줄이면서도 정확한 위치를 추적할 수 있는 SDBTA(Siamese-Dynamic Bayesian Tracking Algorithm)를 제안한다.

## 📎 Related Works

초기 비디오 추적 연구는 SIFT, HOG와 같은 지역적 특징(local features)을 사용하는 특징 기반 방식과, 객체 전체를 하나의 템플릿으로 처리하는 템플릿 매칭 방식으로 나뉜다. 이후 Mean-shift나 MOSSE, MUSTer와 같은 상관 필터(correlation filter) 기반 방법들이 등장하여 효율성을 높였다.

최근에는 딥러닝 모델이 도입되었으며, 특히 GOTURN과 같은 회귀 기반 방식이나 SINT와 같은 Siamese Network 기반 방식이 주목받았다. Siamese Network는 두 이미지의 유사도를 측정하는 능력이 뛰어나 일반 객체 추적(generic object tracking)에서 우수한 성능을 보였다. 하지만 이러한 최신 모델들조차 시스템적인 신뢰 상태(belief states) 생성 능력이나, 불확실성 측정, 그리고 시공간적 정보를 적응적으로 인코딩하여 탐색 범위를 정밀하게 제어하는 능력은 부족한 실정이다.

## 🛠️ Methodology

### 1. Siamese Network (유사도 함수)

본 논문은 일회성 이미지 인식(one-shot image recognition)을 위한 Siamese Network 구조를 채택한다. 이 네트워크는 템플릿 이미지 $z$와 후보 이미지 $x$를 입력받아 두 이미지의 유사도를 반환하는 함수 $f(z, x)$를 학습한다.

- **구조**: 동일한 변환 $\phi$를 두 입력에 적용한 후, 함수 $g$를 통해 최종 유사도 점수를 계산한다.
  $$f(z, x) = g(\phi(z), \phi(x))$$
- **학습**: 5층의 Convolutional Neural Network(CNN) 구조를 사용하며, 정답 라벨 $y \in \{-1, +1\}$와 예측 점수 $v$ 사이의 Logistic Loss를 최소화하도록 학습한다.
  $$l(y, v) = \log(1 + \exp(-yv))$$
- **출력**: 최종 네트워크 출력은 $22 \times 22 \times 128$ 텐서 형태가 된다.

### 2. Dynamic Bayesian Optimization (DOP)

비디오 추적을 다음과 같은 동적 최적화 문제로 정의한다.
$$\text{DOP} = \{ \max f(x, t) \text{ s.t. } x \in F(t) \subseteq S, t \in T \}$$
여기서 $S$는 탐색 공간, $f(x, t)$는 시간 $t$에서 위치 $x$가 타겟 객체와 일치할 때 최대값을 갖는 목적 함수이다.

### 3. Gaussian Process (GP) 및 Surrogate Model

알 수 없는 목적 함수 $f(x, t)$를 근사하기 위해 Gaussian Process Regression (GPR)을 사용한다.

- **Spatio-Temporal GP**: 공간적 상관관계와 시간적 상관관계를 모두 고려하기 위해 다음과 같은 분리 가능한(separable) 공분산 함수(kernel)를 사용한다.
  $$K(\hat{f}(x, t), \hat{f}(x', t')) = K^S(x, x') \cdot K^T(t, t')$$
  여기서 $K^S$는 공간 커널, $K^T$는 시간 커널이며, 실험에서는 Matern 커널을 사용하였다.

### 4. Acquisition Function: MS-EI

어디를 샘플링할지 결정하기 위해 **Memory-Score Expected Improvement (MS-EI)**라는 새로운 획득 함수를 제안한다.
$$\text{MS-EI}(x) = (\mu(x) - f(x^*) - \xi)\Phi(Z) + \sigma(x)\rho(Z)$$
여기서 $Z = \frac{\mu(x) - f(x^*) - \xi}{\sigma(x)}$이며, $\Phi$와 $\rho$는 표준 정규 분포의 PDF와 CDF이다.

특히 $\xi$는 탐색(exploration)과 활용(exploitation)의 균형을 맞추는 파라미터로, 다음과 같이 정의된다.
$$\xi = (\alpha \cdot \text{mean}[f(x)]_D \cdot n^q)^{-1}$$
$\xi$는 일종의 쿨링 스케줄(cooling schedule) 역할을 하여, 탐색 초기에는 새로운 영역을 많이 탐색하도록 유도하고, 샘플링이 진행되어 높은 값들이 발견될수록 탐색 범위를 좁혀 정밀하게 최적점을 찾는 활용 단계로 전이시킨다.

### 5. SDBTA 알고리즘 절차

1. Dynamic GP 모델을 사전 학습한다.
2. 각 프레임 $t$에 대하여:
   - MS-EI를 최대화하는 지점 $\{x_i, t_i\}$를 계산하여 최적의 샘플 위치를 찾는다.
   - 해당 위치의 이미지 패치를 Siamese Network에 쿼리하여 유사도 점수 $y_i$를 얻는다.
   - 새로운 데이터 포인트를 GP 모델에 추가하여 업데이트한다.
   - $d \times d$ (실험에서는 $20 \times 20$) 그리드 상에 GPR 근사치를 렌더링하고, 이를 원래 탐색 공간 크기로 업샘플링(cubic interpolation)한다.
   - 최종적으로 목적 함수가 최대가 되는 위치를 객체의 현재 위치로 업데이트한다.

- **스케일 대응**: 객체의 크기 변화를 처리하기 위해 $\{1.00-p, 1.00, 1.00+p\}$ (여기서 $p=0.05$)의 세 가지 스케일에 대해 유사도 점수를 계산한다.

## 📊 Results

### 실험 설정

- **데이터셋**: VOT14, VOT16의 일부와 CFNET 데이터셋을 사용하였다.
- **비교 대상**: Template Matching (TM), MOSSE, ADNET (강화학습 기반 추적기).
- **평가 지표**: IoU (Intersection over Union).
- **설정**: 프레임당 샘플 수는 80개로 고정하였다.

### 정량적 결과

실험 결과, 제안 방법인 SDBTA가 모든 기준 모델보다 높은 평균 IoU를 기록하였다.

| 모델 | mean IoU | std IoU |
| :--- | :---: | :---: |
| TM | 0.26 | 0.22 |
| MOSSE | 0.10 | 0.25 |
| ADNET | 0.47 | 0.23 |
| **SDBTA (Ours)** | **0.56** | **0.17** |

### 정성적 분석 및 안정성

단순한 수치적 우위 외에도 추적의 안정성 면에서 큰 차이를 보였다. 특정 테스트 비디오('tc boat ce1') 분석 결과, MOSSE는 30프레임 이후, TM은 비디오의 절반 가량, ADNET은 170프레임 이후에 추적에 실패하는 모습을 보였다. 반면, SDBTA는 전 구간에서 안정적인 추적 성능을 유지하였다.

## 🧠 Insights & Discussion

### 강점

본 연구의 가장 큰 강점은 딥러닝 모델의 '표현력'과 베이지안 최적화의 '효율적 탐색 능력'을 성공적으로 결합했다는 점이다. 특히 Gaussian Process를 통해 목적 함수의 불확실성을 모델링하고 MS-EI 획득 함수를 통해 탐색과 활용의 균형을 동적으로 조절함으로써, 적은 수의 쿼리(프레임당 80회)만으로도 정밀한 추적이 가능함을 증명하였다.

### 한계 및 향후 과제

논문에서는 다음과 같은 개선 방향을 제시하고 있다.

1. **다차원 GP 확장**: 현재 공간과 시간만 고려하는 GP를 크기(size) 차원까지 확장하여 5차원 GP로 발전시킬 필요가 있다.
2. **적응적 BO (ABO)**: 학습된 시간 관련 길이-척도(length-scale) 파라미터를 기반으로 탐색 범위와 제약 조건을 적응적으로 변경하는 기법을 도입할 수 있다.
3. **연산 속도 개선**: Fully-convolutional 구조를 통합하여 실시간 이상의 속도를 달성하고, Kronecker inference와 같은 수치 최적화 기법을 통해 GP 연산 효율을 높일 수 있다.

## 📌 TL;DR

본 논문은 **Siamese Network를 유사도 함수로 사용하고 Dynamic Bayesian Optimization(Gaussian Process 기반)을 통해 객체의 위치를 효율적으로 탐색하는 SDBTA 알고리즘을 제안**한다. 이 방식은 딥러닝의 강력한 특징 추출 능력과 통계적 최적화의 효율성을 결합하여, 기존의 딥러닝 추적기나 상관 필터 방식보다 더 높고 안정적인 IoU 성능을 달성하였다. 특히 적은 수의 샘플링만으로도 객체를 정확히 추적할 수 있어, 향후 실시간 감시 시스템이나 고수준 장면 이해 시스템에 적용될 가능성이 높다.
