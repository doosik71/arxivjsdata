# Everything to the Synthetic: Diffusion-driven Test-time Adaptation via Synthetic-Domain Alignment

Jiayi Guo, Junhao Zhao, Chaoqun Du, Yulin Wang, Chunjiang Ge, Zanlin Ni, Shiji Song, Humphrey Shi, Gao Huang (2024)

## 🧩 Problem to Solve

본 논문은 소스 도메인(Source Domain)에서 사전 학습된 모델을 이전에 본 적 없는 변형된 타겟 도메인(Target Domain)에 적용할 때 발생하는 성능 저하 문제를 해결하기 위한 Test-Time Adaptation (TTA) 방법론을 다룬다.

기존의 TTA 방식은 크게 두 가지 방향으로 나뉜다. 첫째, 모델의 가중치를 타겟 데이터 스트림에 맞춰 지속적으로 업데이트하는 전통적인 모델 적응 방식이다. 그러나 이 방식은 타겟 데이터의 양이나 순서에 매우 민감하며, 특히 특정 클래스만 포함된 배치로 학습할 경우 과적합(Overfitting)이 발생할 위험이 크다. 둘째, 최근 등장한 Diffusion 기반 TTA 방식은 모델 가중치 대신 입력 데이터를 적응시키는 방식을 취한다. 이는 사전 학습된 Unconditional Diffusion 모델을 통해 타겟 데이터를 소스 도메인과 유사한 상태로 복원하여 예측하는 방식이다.

하지만 저자들은 Diffusion 기반 TTA에서 복원된 데이터가 시각적으로는 소스 데이터와 구분이 불가능해 보일지라도, 딥러닝 네트워크 관점에서는 여전히 소스 도메인과 정렬되지 않은 '합성 도메인(Synthetic Domain)'에 머물러 있다는 점을 발견하였다. 즉, 타겟 데이터를 소스로 보내려 하지만 실제로는 타겟 $\rightarrow$ 합성 도메인으로 이동하는 것이며, 소스 모델과 합성 도메인 사이의 불일치(Misalignment)가 최종 성능의 병목 현상이 된다는 것이 본 연구가 해결하고자 하는 핵심 문제이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 소스 모델과 타겟 데이터 모두를 동일한 '합성 도메인(Synthetic Domain)'으로 정렬시키는 **Synthetic-Domain Alignment (SDA)** 프레임워크를 제안한 것이다.

중심적인 직관은 타겟 데이터를 소스 도메인으로 억지로 끌어오는 대신, 소스 모델 또한 Diffusion 모델이 생성하는 합성 도메인에 맞게 조정함으로써, 결과적으로 TTA 문제를 '도메인 간 예측'이 아닌 '도메인 내 예측(In-domain prediction)' 문제로 변환하는 것이다. 이를 위해 저자들은 Conditional Diffusion 모델로 라벨링된 데이터를 생성하고, 이를 다시 Unconditional Diffusion 모델로 정렬하여 모델을 미세 조정하는 **Mix of Diffusion (MoD)** 기법을 도입하였다.

## 📎 Related Works

TTA 연구는 크게 모델 적응(Model Adaptation)과 데이터 적응(Data Adaptation)으로 구분된다.

- **모델 적응 방식:** TENT, MEMO 등은 Batch Normalization 통계량을 업데이트하거나 자기지도 학습(Self-supervised learning) 보조 작업을 통해 가중치를 수정한다. 그러나 이러한 방식은 타겟 데이터의 분포가 불균형할 때 성능이 불안정하다는 한계가 있다.
- **데이터 적응 방식:** DiffPure, DDA, GDA 등은 Diffusion 모델의 역과정(Reverse process)을 통해 타겟 데이터를 정제하여 소스 도메인으로 투영한다.

본 논문은 기존 데이터 적응 방식들이 "Target $\rightarrow$ Source" 방향을 지향하지만 실제로는 "Target $\rightarrow$ Synthetic"으로 이동한다는 점을 지적하며, 이에 대응하여 "Source $\rightarrow$ Synthetic" 방향의 모델 적응을 추가함으로써 양측의 도메인을 일치시키는 차별점을 가진다.

## 🛠️ Methodology

SDA 프레임워크는 크게 세 단계의 파이프라인으로 구성된다.

### 1. 소스 도메인 모델 사전 학습 (Source-Domain Model Pretraining)

표준적인 지도 학습 방식을 통해 소스 데이터 $x_{src}^0$로 소스 모델 $f$를 학습시킨다. 이 단계 이후 소스 데이터에 대한 접근은 완전히 차단된다.

### 2. 소스-합성 모델 적응 (Source-to-Synthetic Model Adaptation)

소스 모델 $f$를 합성 도메인 모델 $f'$로 변환하기 위해 **Mix of Diffusion (MoD)** 과정을 거친다.

- **Conditional Diffusion 데이터 생성:** 도메인에 무관한 라벨 집합 $\{y_i\}_{i=1}^K$를 사용하여 Conditional Diffusion 모델 $\varepsilon_\eta^c$를 통해 라벨링된 합성 데이터셋 $\{x_{syn, c}^0, y\}$를 생성한다.
- **Unconditional Diffusion 데이터 정렬:** 생성된 $x_{syn, c}^0$는 Conditional 모델의 도메인에 속해 있으므로, 이를 실제 TTA에 사용될 Unconditional 모델 $\varepsilon_\theta^u$의 도메인으로 맞춘다. 구체적으로, 데이터에 노이즈를 추가하여 $t^*$ 단계까지 보낸 후($x_{syn, c}^{t^*}$), 다시 Unconditional 모델을 통해 디노이징(Denoising)하여 $x_{syn, u}^0$를 얻는다.
- **모델 미세 조정:** 이렇게 정렬된 합성 데이터셋 $\{x_{syn, u}^0, y\}$를 사용하여 소스 모델 $f$를 미세 조정하여 $f'_u$를 생성한다.

### 3. 타겟-합성 데이터 적응 (Target-to-Synthetic Data Adaptation)

테스트 시점에 입력된 타겟 데이터 $x_{trg}^0$를 Unconditional Diffusion 모델 $\varepsilon_\theta^u$를 이용하여 합성 도메인 데이터 $x_{syn, u}^0$로 변환한다. 이때 DDA에서 제안된 구조적 가이드(Structure guidance)를 사용하여 이미지의 시맨틱 내용을 보존한다.

### 4. 추론 및 앙상블 (Inference & Ensembling)

최종 예측은 원본 소스 모델 $f$가 타겟 데이터 $x_{trg}^0$를 예측한 결과와, 적응된 합성 모델 $f'_u$가 합성 데이터 $x_{syn, u}^0$를 예측한 결과의 앙상블로 수행된다.
$$\hat{y} = \arg \max_{y} (q(y|x_{trg}^0) + q'(y|x_{syn, u}^0))$$
여기서 $q(\cdot)$와 $q'(\cdot)$는 각각 소스 모델과 합성 도메인 모델의 출력 분포이다.

## 📊 Results

### 실험 설정

- **데이터셋:** ImageNet-C (부패 데이터), ImageNet-W (워터마크 데이터), CIFAR-10-C, PASCAL VOC-C.
- **모델 아키텍처:** ResNet, Swin Transformer, ConvNeXt, 그리고 MLLM인 LLaVA-1.5-7b.
- **비교 대상:** Source(기본), MEMO, DiffPure, GDA, DDA.

### 주요 결과

- **분류 작업 (ImageNet-C/W):** SDA는 모든 모델 아키텍처에서 기존 SOTA 방법론인 DDA 및 GDA보다 우수한 성능을 보였다. ImageNet-C에서 DDA 대비 약 $2.5\% \sim 2.9\%$의 정확도 향상을 기록하였다.
- **확장성 검증:**
  - **Small Dataset:** CIFAR-10-C에서 DDA 대비 평균 $7.1\%$의 성능 향상을 보였다.
  - **Semantic Segmentation:** PASCAL VOC-C에서 mIOU 기준 DDA보다 $1.2\%$ 향상된 결과를 얻었다.
  - **MLLM (LLaVA):** VQA 기반 분류 작업에서 SDA를 적용했을 때, 단순 소스 데이터 미세 조정이나 DDA보다 $2.4\%$ 높은 성능을 달성하였다.
- **데이터 스트림 민감도:** UniTTA 벤치마크를 통해 데이터의 순서나 클래스 불균형 상황을 테스트한 결과, Diffusion 기반 방식들이 전통적인 TTA보다 훨씬 안정적이었으며, 그중 SDA가 가장 높은 성능을 유지하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문의 가장 큰 통찰은 **"시각적 유사성이 도메인 정렬을 의미하지 않는다"**는 점이다. 저자들은 Grad-CAM 시각화를 통해, DDA(데이터만 적응)를 적용했을 때 모델이 여전히 엉뚱한 영역을 활성화하는 반면, SDA(모델과 데이터 모두 적응)를 적용했을 때는 소스 모델이 소스 데이터를 처리할 때와 매우 유사한 활성화 맵을 생성함을 보여주었다. 이는 SDA가 실제로 딥러닝 모델의 내부 피처 공간에서 도메인 정렬을 성공적으로 수행했음을 시사한다.

### 한계 및 비판적 해석

- **합성 데이터 생성 비용:** 모델 적응 단계에서 대량의 합성 데이터를 생성하고 미세 조정하는 과정이 필요하다. 비록 이 과정이 테스트 타임 이전에 한 번만 수행되면 되지만, Diffusion 모델의 생성 속도와 자원 소모는 여전히 부담이 될 수 있다.
- **가정 사항:** 본 논문은 Conditional Diffusion 모델과 Unconditional Diffusion 모델이 모두 소스 도메인에서 사전 학습되었다고 가정한다. 만약 적절한 사전 학습된 생성 모델을 구할 수 없는 특수 도메인의 경우 적용이 어려울 수 있다.
- **데이터 양의 영향:** 실험 결과 $N=1K$장의 이미지(클래스당 1장)만으로도 DDA보다 성능이 좋았다는 점은, 모델이 클래스의 지식을 배우는 것이 아니라 합성 도메인의 '특성' 자체에 적응하는 것임을 보여준다. 이는 미세 조정의 목적이 지식 습득이 아닌 도메인 정렬에 있음을 명확히 한다.

## 📌 TL;DR

본 논문은 Diffusion 기반 TTA에서 발생하는 '소스 도메인'과 '합성 도메인' 간의 불일치 문제를 해결하기 위해, 소스 모델과 타겟 데이터를 모두 동일한 합성 도메인으로 정렬시키는 **SDA(Synthetic-Domain Alignment)** 프레임워크를 제안한다. **Mix of Diffusion (MoD)** 기법을 통해 모델을 합성 도메인에 맞게 미세 조정함으로써, 복잡한 cross-domain TTA 문제를 단순한 in-domain 예측 문제로 변환하였다. 이 방법은 분류, 세그멘테이션, 그리고 MLLM에 이르기까지 폭넓은 확장성을 보였으며, 특히 데이터 스트림의 불균형에 영향을 받지 않는 강건한 성능을 입증하였다.
