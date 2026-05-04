# Optimization and Deployment of Deep Neural Networks for PPG-based Blood Pressure Estimation Targeting Low-power Wearables

Alessio Burrello, Francesco Carlucci, Giovanni Pollo, Xiaying Wang, Massimo Poncino, Enrico Macii, Luca Benini, Daniele Jahier Pagliari (2024)

## 🧩 Problem to Solve

본 연구는 웨어러블 기기와 같은 저전력 장치에서 광혈류측정(Photoplethysmography, PPG) 신호를 이용하여 혈압(Blood Pressure, BP)을 추정하는 문제를 해결하고자 한다. 혈압의 지속적인 모니터링은 고혈압, 심근병증, 심부전과 같은 심혈관 질환 예방에 매우 중요하며, 스마트워치와 같은 비침습적 웨어러블 기술은 비용 효율적이고 일상생활에 지장을 주지 않는 모니터링 솔루션을 제공할 수 있다.

최근의 딥러닝 모델(DNN)은 기존의 전통적인 방식보다 높은 성능을 보이지만, 이러한 모델들은 일반적으로 방대한 파라미터 저장 공간과 높은 연산 비용을 요구한다. 이는 메모리 용량이 제한적이고 전력 소모 및 지연 시간(latency)에 민감한 웨어러블 장치에 그대로 배포하기에는 부적합하다는 문제가 있다. 따라서 본 논문의 목표는 정확도를 유지하면서도 모델 크기를 획기적으로 줄여, 초저전력 SoC인 GAP8에 배포 가능한 경량화된 DNN 설계 파이프라인을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 하드웨어 인식 Neural Architecture Search (NAS)와 양자화(Quantization)를 결합한 완전 자동화된 DNN 설계 파이프라인을 제안하고, 이를 실제 저전력 하드웨어에 구현하여 검증했다는 점이다.

중심적인 아이디어는 기존의 최첨단(State-of-the-art, SoA) 모델을 Seed 모델로 설정하고, 이를 기반으로 하드웨어 비용(모델 크기)과 예측 오차 사이의 최적의 균형점을 찾는 파레토 최적(Pareto-optimal) 구조를 자동으로 탐색하는 것이다. 이를 통해 메모리 제약이 심한 GAP8 환경에서도 구동 가능하면서도, 기존 SoA 모델보다 더 작거나 더 정확한 모델을 도출하였다.

## 📎 Related Works

혈압 추정을 위해 Random Forest (RF)나 Support Vector Regression (SVR)과 같은 고전적인 머신러닝 방법들이 사용되어 왔으며, 최근에는 1D Convolutional Neural Networks (CNN) 기반의 DNN 모델들이 주를 이루고 있다. DNN 접근 방식은 크게 두 가지로 나뉜다. 첫째는 PPG 신호 윈도우를 입력받아 단일 스칼라 값(SBP 또는 DBP)을 예측하는 회귀(Regression) 방식(예: ResNet 기반)이며, 둘째는 PPG 신호를 통해 BP 전체 시계열을 재구성하는 신호-대-신호(Signal-to-Signal, sig2sig) 방식(예: UNet 기반)이다.

DNN은 특성 추출(Feature Extraction) 과정이 필요 없고 일반화 능력이 뛰어나지만, 앞서 언급한 대로 파라미터 수가 많아 리소스가 제한된 웨어러블 기기에 배포하기 어렵다는 한계가 있다. 본 연구는 이러한 하드웨어 제약 조건을 설계 단계에서부터 고려하는 cost-aware NAS를 적용함으로써 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인 구조

제안된 최적화 흐름은 **[Seed 모델 선택 $\rightarrow$ Gradient-based NAS $\rightarrow$ Quantization $\rightarrow$ Hardware Deployment]**의 순서로 진행된다.

### 1. Neural Architecture Search (NAS)

본 연구는 SuperNet 기반의 gradient-based NAS를 사용한다.

- **작동 원리**: Seed 모델의 각 합성곱(Convolutional) 레이어를 표준 1D Conv(C), Depthwise-separable block(DW), 그리고 Identity operation(ID)으로 구성된 후보군(Pool)으로 대체한다.
- **레이어 선택**: 각 후보군의 출력은 학습 가능한 파라미터 $\theta_i$에 의해 소프트맥스(Softmax) 가중치 합으로 결정된다.
- **손실 함수**: 다음의 수식을 통해 작업 정확도와 모델 비용을 동시에 최적화한다.
$$\mathcal{L}_{W, \theta} = \mathcal{L}_{task} + \lambda \mathcal{R}_{\theta}$$
여기서 $\mathcal{L}_{task}$는 예측값과 실제값 사이의 Mean Squared Error (MSE)이며, $\mathcal{R}_{\theta}$는 모델의 크기(파라미터 수)를 나타내는 비용 함수이다. $\lambda$ 값을 조절함으로써 정확도와 크기 사이의 다양한 트레이드-오프를 가진 파레토 최적 모델들을 생성할 수 있다.

### 2. Quantization (양자화)

NAS를 통해 선정된 모델들을 `int8` 정밀도로 양자화하여 메모리 점유율과 에너지 소모를 더욱 줄인다.

- **방법론**: Quantization-Aware Training (QAT)을 적용하며, 가중치에는 min-max affine 양자화를, 레이어의 입력 및 출력에는 Parametrized Clipping Activation (PaCT) 방식을 사용한다.
- **정밀도**: 가중치와 활성화 함수는 8비트로 처리하며, 누적 연산 및 바이어스는 32비트로 처리한다.

### 3. Hardware Deployment (하드웨어 배포)

최종 최적화된 모델은 RISC-V 기반의 초저전력 SoC인 GAP8에 배포된다.

- **GAP8 구조**: 8개의 일반 목적 코어 클러스터와 512kB의 메인 메모리, 64kB의 L2 캐시를 가진다.
- **배포 도구**: DORY 컴파일러를 사용하여 양자화된 DNN을 최적화된 C 코드로 자동 변환하여 배포한다.

## 📊 Results

### 실험 설정

- **데이터셋**: PPGBP, BCG, Sensors, UCI 등 4개의 공개 데이터셋을 사용하였으며, 모두 125Hz로 리샘플링되었다.
- **평가 지표**: 수축기 혈압(SBP)과 이완기 혈압(DBP)에 대해 각각 Mean Absolute Error (MAE)를 측정하였다.
- **비교 대상**: 기존 SoA 모델(ResNet, UNet), 고전적 ML 모델(RF, SVR)과 비교하였다.

### 주요 결과

1. **파레토 분석**: 모든 데이터셋에서 NAS를 통해 도출된 모델들이 Seed 모델보다 적은 파라미터로 유사하거나 더 낮은 MAE를 달성하였다. 특히 BCG 데이터셋에서는 UNet 기반 NAS 모델이 Seed ResNet보다 MAE를 4.7%~6.7% 개선함과 동시에 파라미터 수를 3.8배 줄였다.
2. **데이터셋 규모에 따른 성능**: 데이터셋 규모가 커질수록(특히 UCI) DNN의 우위가 뚜렷해졌다. UCI 데이터셋에서 NAS 모델은 SBP MAE 16.655 mmHg, DBP MAE 7.86 mmHg를 기록하며 고전적 ML 모델들을 능가하였다.
3. **하드웨어 배포 결과 (UCI 데이터셋 기준)**:
    - **메모리**: Seed ResNet은 너무 커서 GAP8에 배포가 불가능했지만, 모든 NAS 최적화 모델은 배포 가능했다.
    - **성능**: `Resnet-S` 모델은 DBP MAE 8.08 mmHg를 달성했으며, 에너지 소모는 0.37 mJ에 불과했다. `UNet-S` 모델은 SBP MAE 17.2 mmHg, 지연 시간 8.91 ms, 에너지 소모 0.45 mJ를 기록하였다.

## 🧠 Insights & Discussion

본 연구는 저전력 임베디드 환경에서 딥러닝 모델을 구현하기 위해 단순한 경량화 기법이 아닌, 아키텍처 탐색(NAS)과 양자화를 통합한 파이프라인이 매우 효과적임을 보여주었다.

**주요 통찰:**

- **네트워크 구조의 영향**: BCG와 Sensors와 같은 상대적으로 작은 데이터셋에서는 UNet 구조가 더 우수한 성능을 보였다. 이는 시계열 전체를 재구성하는 `sig2sig` 작업이 더 풍부한 학습 신호를 제공하기 때문으로 분석된다. 반면, 매우 큰 데이터셋인 UCI에서는 ResNet 기반 모델이 더 높은 성능을 나타냈다.
- **양자화의 영향**: `fp32`에서 `int8`로 양자화할 때 MAE가 최대 9.8%까지 상승하는 성능 저하가 발생했으며, 특히 ResNet 구조가 UNet보다 양자화에 더 민감한 경향을 보였다.
- **실용적 한계**: 본 논문에서 구현한 교차 환자(cross-patient) 추론 방식은 의료 기기 수준의 정밀도 요구사항을 충족하기에는 오차가 크며, 이를 위해서는 환자 맞춤형 미세 조정(personalized fine-tuning)이 추가로 필요함을 명시하고 있다.

## 📌 TL;DR

본 논문은 PPG 신호를 이용한 혈압 추정 DNN을 초저전력 웨어러블 기기에 배포하기 위해, **하드웨어 인식 NAS와 `int8` 양자화를 결합한 자동 최적화 파이프라인**을 제안하였다. 이를 통해 기존 SoA 모델 대비 정확도를 유지하면서 크기를 최대 73.36% 줄였으며, 실제 GAP8 SoC 상에서 0.37 mJ의 매우 낮은 에너지 소모로 실시간 추론이 가능함을 증명하였다. 이 연구는 리소스 제약이 극심한 엣지 디바이스에서 고성능 생체 신호 처리 모델을 구현하는 실질적인 방법론을 제시했다는 점에서 향후 웨어러블 헬스케어 기기 설계에 중요한 역할을 할 것으로 보인다.
