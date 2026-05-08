# Self-Supervised Out-of-Distribution Detection in Brain CT Scans

Abinav Ravi Venkatakrishnan, Seong Tae Kim, Rami Eisawy, Franz Pfister, Nassir Navab (2020)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 의료 영상 데이터, 특히 3D 뇌 CT 스캔 데이터의 어노테이션(Annotation) 부족 문제이다. 3D 의료 데이터를 정밀하게 라벨링하는 작업은 매우 많은 시간과 비용이 소모되는 작업이다. 또한, 설령 라벨링된 데이터가 존재하더라도 데이터 불균형(Data Imbalance) 문제가 심각하다. 대다수의 스캔 데이터는 정상인에게서 얻어지며, 비정상 사례는 수가 적을 뿐만 아니라 그 양상(Intraclass variation)이 매우 다양하기 때문에 지도 학습(Supervised Learning) 기반의 접근 방식으로는 한계가 있다.

따라서 본 논문의 목표는 정상 스캔 데이터만을 사용하여 모델을 학습시키고, 분포 외(Out-of-Distribution, OOD) 데이터인 비정상 스캔을 효과적으로 감지할 수 있는 자가 지도 학습(Self-Supervised Learning, SSL) 기반의 이상치 탐지(Anomaly Detection) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **국소적인 픽셀 수준의 정보(Local fine-grained information)**와 **전역적인 이미지 수준의 기하학적 문맥(Global context)**을 동시에 활용하여 이상치 점수를 계산하는 것이다.

구체적으로는 다음의 두 가지 설계를 통해 이를 달성한다.

1. **Reconstruction-based module**: Variational Autoencoder(VAE)를 통해 정상 데이터의 분포를 학습하고, 이를 재구성함으로써 국소적인 이상 부위를 탐지한다.
2. **Geometric transformation predictor**: 이미지에 적용된 기하학적 변환(회전, 이동)을 예측하게 함으로써 모델이 정상 스캔의 전역적인 구조와 특성을 더 잘 학습하도록 유도한다.

또한, 모델이 복잡한 의료 영상의 일반적인 특징을 더 잘 추출할 수 있도록 **Context Restoration(문맥 복원)** 기법을 이용한 사전 학습(Pretraining) 단계를 도입하였다.

## 📎 Related Works

기존의 비지도 이상치 탐지 연구들은 주로 Autoencoder 기반의 재구성 방법이나 Variational Autoencoder(VAE)를 사용해 왔다. 이러한 방식들은 정상 데이터로만 모델을 학습시킨 뒤, 비정상 데이터가 입력되었을 때 재구성 오차(Reconstruction Error)가 크게 발생하는 점을 이용하여 이상치를 탐지한다.

본 논문은 이러한 기존 방식들이 국소적인 재구성 오차에만 의존한다는 점을 보완하고자 한다. 기존 연구들이 주로 세그멘테이션(Segmentation) 관점에서 이상치를 탐지하려 했다면, 본 연구는 자가 지도 학습을 통해 분류(Classification) 성능을 높이는 방향으로 확장하여 전역적 기하학적 정보를 통합했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Pretraining with Context Restoration

모델이 뇌 CT 영상의 일반적인 특징을 학습할 수 있도록 사전 학습을 수행한다. 입력 이미지의 일부 패치(Patch)를 서로 바꾸어 섞은(Swapped) 이미지 $\hat{x}_i$를 생성하고, 이를 원래 이미지 $x_i$로 복원하도록 VAE를 학습시킨다. 이때 사용되는 손실 함수는 다음과 같다.

$$L_{cr} = \|x_i - f(\hat{x}_i)\|^2$$

여기서 $f$는 VAE 함수를 의미하며, 이 과정을 통해 모델은 이미지의 구조적 문맥을 이해하게 된다.

### 2. Multi-Task Learning

사전 학습 이후, 모델은 재구성과 기하학적 변환 예측이라는 두 가지 작업을 동시에 수행하는 Multi-task 학습을 진행한다.

**A. Geometric Transformation Predictor**
입력 이미지에 무작위로 회전($0^\circ, 90^\circ, 180^\circ, 270^\circ$) 또는 수직/수평 이동($1/8$ 이미지 크기)을 적용하고, 모델이 어떤 변환이 적용되었는지 예측하게 한다. 이 작업의 손실 함수는 Cross-entropy loss를 사용한다.

$$L_{geo} = -\sum_{i}^{N} q_i \log g(\tilde{x}_{q_i})$$

여기서 $g$는 변환 예측 함수, $q_i$는 실제 적용된 변환 라벨, $\tilde{x}_{q_i}$는 변환된 입력 이미지이다.

**B. VAE Fine-tuning 및 전체 손실 함수**
VAE는 변환된 이미지들을 다시 원래대로 재구성하도록 미세 조정(Fine-tuning)된다. 전체 학습을 위한 최종 손실 함수는 다음과 같이 정의된다.

$$L_{multitask} = L_{geo} + \epsilon L_{rec}$$

여기서 $L_{rec} = \|x_i - f(x_i)\|^2$이며, $\epsilon$은 두 손실 항의 균형을 맞추는 스케일링 인자이다.

### 3. Anomaly Score Calculation

테스트 단계에서 이상치 점수(Anomaly Score)는 기하학적 변환 예측 결과와 재구성 결과의 가중 평균으로 계산된다.

$$\text{score} = (1 - \lambda)s_g + \lambda s_r$$

- $s_g$: 기하학적 변환 예측기의 Softmax 점수에서 유도된 점수로, 가능한 변환 조합들에 대한 예측 오차의 평균값이다.
- $s_r$: 재구성 오차에 기반한 점수로, $s_r = \alpha \times \|x_i - f(x_i)\|^2$로 정의된다.
- $\lambda$: 하이퍼파라미터이며, 본 실험에서는 $0.5$로 설정되었다.

## 📊 Results

### 실험 설정

- **데이터셋**: 임상 뇌 CT 스캔 데이터를 사용하였으며, 학습에는 오직 **정상 슬라이스(Normal slices)**만 사용하였다.
- **테스트 셋**: 위축(Atrophy), 뇌출혈(Intracranial bleeding), 허혈(Ischemia), 해면상 혈관종(Cavernoma), 동맥류(Aneurysm), 종양(Tumor) 등 다양한 비정상 사례가 포함되었다.
- **평가 지표**: 분류 성능 측정을 위해 AUROC와 AUPR을 사용하였고, 국소화 성능 측정을 위해 Dice Similarity Coefficient(DSC)를 사용하였다.

### 정량적 결과

비교 실험 결과, 제안된 Multi-task Framework가 기존 방법론들보다 우수한 분류 성능을 보였다.

| Method | AUROC | AUPR | DSC |
| :--- | :---: | :---: | :---: |
| VAE [7] | 0.668 | 0.704 | $0.110 \pm 0.021$ |
| Context-encoding VAE [3] | 0.766 | 0.640 | $0.112 \pm 0.021$ |
| VAE with Pretraining (ours) | 0.673 | 0.772 | $0.112 \pm 0.021$ |
| **Multi-task Framework (ours)** | **0.822** | **0.868** | $0.086 \pm 0.024$ |

### 결과 해석

- **분류 성능 향상**: Context Restoration 사전 학습을 적용한 VAE가 기본 VAE보다 성능이 높았으며, 여기에 기하학적 변환 예측을 결합한 Multi-task Framework가 가장 높은 AUROC(0.822)와 AUPR(0.868)을 기록하였다.
- **세그멘테이션 성능 저하**: 반면, DSC 결과에서는 제안 방법이 다른 모델들보다 약간 낮은 성능을 보였다. 이는 인코더가 재구성과 변환 예측이라는 두 가지 목표를 동시에 학습하면서, 세그멘테이션의 핵심인 재구성 성능이 일부 희생되었기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 연구는 뇌 CT 스캔의 이상치 탐지에서 전역적 문맥(Global context)을 학습하는 것이 분류 성능 향상에 결정적인 역할을 한다는 것을 입증하였다. 특히 자가 지도 학습(SSL) 기반의 사전 학습과 멀티태스크 학습이 의료 영상의 복잡한 특징을 추출하는 데 효과적임을 보여주었다.

가장 주목할 점은 **분류 성능(Classification)**과 **세그멘테이션 성능(Localization/Segmentation)** 사이의 트레이드-오프(Trade-off) 관계가 존재한다는 것이다. 전역적 특징을 학습하여 "이 이미지가 이상한가"를 판단하는 능력은 향상되었으나, "어디가 이상한가"를 픽셀 단위로 짚어내는 정밀도는 다소 하락하였다. 따라서 실제 적용 시에는 사용 목적이 단순 스크리닝(분류)인지, 정밀 진단(위치 탐지)인지에 따라 적절한 모델 선택이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 뇌 CT 영상에서 라벨링 부족 문제를 해결하기 위해 **Context Restoration 사전 학습**과 **기하학적 변환 예측**을 결합한 자가 지도 학습 기반의 이상치 탐지 프레임워크를 제안한다. 전역적 구조 정보와 국소적 재구성 정보를 동시에 활용함으로써 기존 VAE 기반 방식보다 우수한 이상치 분류 성능(AUROC 0.822)을 달성하였으며, 이는 향후 의료 영상의 비지도 진단 시스템 연구에 새로운 방향성을 제시한다.
