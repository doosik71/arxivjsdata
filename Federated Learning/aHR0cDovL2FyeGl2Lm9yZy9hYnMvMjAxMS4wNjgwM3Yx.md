# Federated Learning System without Model Sharing through Integration of Dimensional Reduced Data Representations

Anna Bogdanova, Akie Nakai, Yukihiko Okada, Akira Imakura and Tetsuya Sakurai (2020)

## 🧩 Problem to Solve

본 논문은 기존 연합 학습(Federated Learning) 시스템의 핵심적인 한계점인 **'모델 공유(Model Sharing)'** 문제를 해결하고자 한다. 일반적으로 Federated Averaging(FedAvg)과 같은 연합 학습 방식은 중앙 서버와 참여자 간에 모델 가중치(Weights)를 반복적으로 주고받으며 모델을 업데이트한다. 그러나 이러한 방식은 다음과 같은 심각한 문제점과 제약 사항을 갖는다.

1.  **보안 및 프라이버시 취약성**: 모델 가중치가 공유되는 과정에서 모델 인버전 공격(Model Inversion Attack)이나 모델 포이즈닝 공격(Model Poisoning Attack)과 같은 보안 위협에 노출될 수 있다.
2.  **기업의 지적 재산권 문제**: 민간 부문의 협업 시, 학습된 모델 자체는 기업의 핵심 노하우나 비용이 많이 투입된 자산이므로, 이를 타사와 공유하는 것에 대해 강한 거부감이 존재하여 협업의 동기가 저하된다.
3.  **인프라 및 통신 제약**: 모델을 공유하고 반복적으로 업데이트하는 과정은 높은 통신 비용을 발생시키며, 보안상의 이유로 잦은 통신이 불가능한 환경에서는 적용하기 어렵다.

따라서 본 연구의 목표는 모델을 공유하지 않고도 여러 참여자의 분산된 데이터를 통합하여 학습 효과를 얻을 수 있는 새로운 연합 학습 프레임워크인 **Data Collaboration**의 실용성을 분석하고 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **차원 축소(Dimensionality Reduction)**를 통해 데이터의 핵심 특징만을 추출한 **중간 표현(Intermediate Representation, IR)**을 생성하고, 이를 통합하여 분석하는 것이다.

중심적인 설계 직관은 각 참여자가 자신만의 비밀스러운 차원 축소 함수 $f_i$를 사용하여 데이터를 변환하면, 서버는 원본 데이터를 복원할 수 없지만, 통합된 저차원 공간에서의 데이터 분포는 학습에 활용할 수 있다는 점이다. 특히, **앵커 데이터(Anchor Data)**라는 공통의 가상 데이터를 도입하여 서로 다른 참여자가 생성한 서로 다른 IR들을 하나의 공통된 공간으로 투영(Projection)시킴으로써, 모델 공유 없이도 데이터 수준의 협업을 가능하게 했다.

## 📎 Related Works

기존의 데이터 분석 방식과 연합 학습의 한계는 다음과 같다.

-   **데이터 익명화(Data Anonymization)**: 개인 식별자를 제거하거나 데이터를 섭동(Perturbation)시키는 방식은 최근 연구들에 의해 재식별 가능성이 높다는 점이 밝혀져 안전하지 않은 것으로 평가된다.
-   **Federated Learning (FedAvg)**: Google에서 제안한 FedAvg는 통신 효율성이 높고 일부 사용자의 이탈에도 견고하다는 장점이 있다. 이를 보완하기 위해 Secure Aggregation이나 Differential Privacy 등의 기법이 추가되었으나, 여전히 참여자들이 최종적으로 '공유된 하나의 모델'을 갖게 된다는 점은 변하지 않는다.

본 논문에서 제안하는 **Data Collaboration** 방식은 모델 가중치를 공유하는 대신, 로컬에서 생성된 저차원 표현물을 통합하는 방식을 취함으로써 기존 FL 시스템과의 차별점을 갖는다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
Data Collaboration 시스템은 $d$명의 참여자와 하나의 semi-trusted 서버로 구성된다. 전체 프로세스는 [로컬 차원 축소 $\rightarrow$ 앵커 데이터 공유 $\rightarrow$ 서버에서의 통합 표현 생성 $\rightarrow$ 로컬 모델 학습]의 순서로 진행된다.

### 2. 상세 절차 및 방정식 설명

**가. 로컬 차원 축소 (User Side)**
각 참여자 $i$는 자신의 원본 데이터 $X_i \in \mathbb{R}^{n_i \times m}$에 대해 로컬 차원 축소 함수 $f_i$를 적용하여 중간 표현(IR) $\tilde{X}_i$를 생성한다.
$$\tilde{X}_i = f_i(X_i) \in \mathbb{R}^{n_i \times l_i} \quad (0 < l_i \le m)$$
이때 $f_i$는 각 참여자가 독자적으로 결정하며 외부로 공개하지 않는다. 따라서 동일한 데이터 $x$에 대해서도 참여자마다 결과가 다르다 ($f_i(x) \neq f_j(x)$).

**나. 앵커 데이터를 통한 통합 (Server Side)**
서버는 서로 다른 $\tilde{X}_i$들을 하나의 공간으로 모으기 위해 변환 함수 $g_i$를 찾는다. 이를 위해 참여자들은 공통의 앵커 데이터 $X^{anc} \in \mathbb{R}^{r \times m}$를 생성하고, 각자 $f_i$를 적용한 $\tilde{X}^{anc}_i$를 서버에 전달한다.
서버는 모든 $\tilde{X}^{anc}_i$가 공통의 표현 $Z$에 가깝게 투영되도록 하는 $g_i$를 최적화한다.
$$\min_{G_1, \dots, G_d, Z} \sum_{i=1}^{d} \| Z - g_i(\tilde{X}^{anc}_i) \|_F^2$$

**다. 선형 변환 및 SVD 해결**
$g_i$가 선형 변환($\hat{X}_i = \tilde{X}_i G_i$)인 경우, 이 문제는 특이값 분해(SVD)를 통해 해결할 수 있다. 서버는 통합 앵커 행렬 $\tilde{X}^{anc} = [\tilde{X}^{anc}_1, \dots, \tilde{X}^{anc}_d]$를 SVD 하여 다음과 같이 분해한다.
$$\tilde{X}^{anc} = [U_1, U_2] \begin{bmatrix} \Sigma_1 & 0 \\ 0 & \Sigma_2 \end{bmatrix} [V_1^T, V_2^T]$$
여기서 $Z$를 $U_1$으로 설정하면, 각 사용자의 변환 행렬 $G_i$는 다음과 같이 계산된다.
$$G_i = (\tilde{X}^{anc}_i)^\dagger U_1$$
(여기서 $\dagger$는 pseudo-inverse를 의미한다.)

**라. 최종 학습 (User Side)**
서버는 변환된 데이터 $\hat{X}_i = \tilde{X}_i G_i$와 레이블 $L$을 다시 사용자에게 전달한다. 사용자는 이제 통합된 데이터셋 $\hat{X}$를 사용하여 자신만의 모델 $h$를 독립적으로 학습시킨다.

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: MNIST, Fashion-MNIST (28x28 그레이스케일 이미지, 10 클래스).
-   **비교 대상**: Centralized Learning (상한선), Individual Learning (하한선), Federated Averaging (FedAvg).
-   **모델**: 2개의 은닉층(512, 128 노드)을 가진 Fully-connected Neural Network.
-   **설정**: 
    -   **Type 1**: 참여자 수 증가 (1명 $\rightarrow$ 10명), 인당 데이터 100개.
    -   **Type 2**: 참여자 수 고정 (5명), 인당 데이터 양 증가 (100개 $\rightarrow$ 1000개).

### 2. 주요 결과
-   **파라미터 영향**: IR의 차원이 높을수록, 그리고 공유되는 앵커 데이터의 양이 많을수록 분류 정확도가 향상됨을 확인하였다.
-   **MNIST 결과**: DC와 FedAvg 모두 참여자 수가 증가함에 따라 Centralized Learning 성능에 근접하였다. 참여자가 7명 이상일 때는 FedAvg가 약간 더 우세했다.
-   **Fashion-MNIST 결과**: 
    -   **Type 1**: DC가 전 구간에서 FedAvg보다 우수한 성능을 보였으며, 특히 참여자 수가 적은 설정에서 DC의 효율성이 매우 높았다.
    -   **Type 2**: 데이터 양이 증가함에 따라 FedAvg는 불안정한 동작을 보인 반면, DC는 안정적으로 성능이 향상되어 Centralized Learning에 근접하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
본 연구는 모델 공유 없이 데이터 표현의 통합만으로도 기존의 연합 학습(FedAvg)과 유사하거나, 특정 상황(소규모 참여자, 복잡한 데이터셋)에서는 더 뛰어난 성능을 낼 수 있음을 입증하였다. 이는 차원 축소 과정이 일종의 효과적인 특징 추출(Feature Extraction) 역할을 수행하여, 적은 양의 데이터로도 효율적인 학습을 가능하게 했기 때문으로 분석된다.

### 2. 한계 및 비판적 해석
-   **프라이버시 보장**: 본 논문은 원본 데이터의 복구가 어렵다는 점을 들어 프라이버시를 주장하지만, 수학적으로 엄밀한 프라이버시 보장(Formal Privacy Guarantee)은 제공하지 않는다. 통계적 프라이버시 보장에 대한 추가 연구가 필요하다.
-   **모델 확장성**: 현재 실험은 단순한 MLP 모델에 한정되어 있다. CNN과 같은 복잡한 아키텍처나 텍스트, 강화학습 등 다른 도메인에 적용 가능한 IR 방법론에 대한 연구가 미비하다.

### 3. 실용적 시나리오 제안
저자들은 협업의 성격에 따라 두 가지 시나리오를 제시한다.
-   **서비스 제공자 협업 (Service Provider Collaboration)**: 단일 서비스 품질 유지가 중요하므로 모델 공유 방식(FedAvg)이 적합하다.
-   **경쟁사 간 협업 (Competitor Collaboration)**: 은행, 보험, 의료 분야처럼 데이터 협력의 이득은 얻되 자신의 예측 모델(노하우)은 숨기고 싶은 경우, 본 논문이 제안하는 **Data Collaboration** 방식이 강력한 인센티브를 제공하며 구현이 더 쉬울 것이다.

## 📌 TL;DR

본 논문은 모델 가중치를 공유하지 않는 새로운 연합 학습 프레임워크인 **Data Collaboration**을 제안한다. 각 참여자가 데이터를 저차원으로 축소하고, 앵커 데이터를 이용해 서버에서 이를 하나의 공통 공간으로 통합함으로써, 원본 데이터와 모델을 보호하면서도 협업 학습의 이점을 누릴 수 있게 한다. 실험 결과, 특히 참여자 수가 적거나 데이터 복잡도가 높은 경우 기존 FedAvg보다 우수한 성능을 보였으며, 이는 기업 간의 전략적 데이터 협력(Competitor Collaboration) 시나리오에서 매우 유용한 대안이 될 수 있다.