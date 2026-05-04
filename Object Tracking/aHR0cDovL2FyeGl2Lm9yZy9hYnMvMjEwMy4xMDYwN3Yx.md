# DCF-ASN: Coarse-to-fine Real-time Visual Tracking via Discriminative Correlation Filter and Attentional Siamese Network

Xizhe Xue, Ying Li, Xiaoyue Yin, Qiang Shen (2020)

## 🧩 Problem to Solve

본 논문은 비주얼 트래킹(Visual Tracking) 분야에서 계산 효율성이 뛰어난 Discriminative Correlation Filter(DCF)의 장점과 강력한 특징 표현력을 가진 Siamese Network의 장점을 결합하고자 한다. 

기존의 Siamese 기반 트래커들은 높은 정확도를 보이지만, 온라인 업데이트(Online-updating)를 적용한 경우 심각한 방해 요소가 발생했을 때 누적된 오류로 인해 타겟을 완전히 놓치는 Model Drift 현상이 발생할 가능성이 크다. 반면, DCF 기반 트래커들은 매우 빠른 속도를 자랑하지만 딥러닝 네트워크의 잠재력을 완전히 활용하지 못하는 한계가 있다. 따라서 본 연구의 목표는 DCF의 실시간 응답성과 Siamese 네트워크의 정밀한 위치 추정 능력을 통합하여, 정확도와 속도 사이의 균형을 맞춘 Coarse-to-fine 트래킹 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Coarse-to-fine 전략**을 도입하여 타겟의 상태를 단계적으로 추론하는 것이다.

1.  **Coarse-to-fine 프레임워크**: 온라인으로 업데이트되는 DCF 모듈이 타겟의 위치와 크기를 대략적으로(Coarse) 예측하면, 오프라인으로 학습된 Asymmetric Siamese Network(ASN)가 이를 가이드 삼아 타겟의 위치를 정밀하게(Fine) 보정하는 구조를 제안한다.
2.  **Attentional Siamese Network (ASN)**: 템플릿 이미지로부터 적절한 채널 가중치(Channel weights)를 학습하는 Attention subnet과, 이를 이용해 제안된 영역(Proposal)의 점수를 예측하는 Estimation subnet으로 구성된 비대칭 구조의 네트워크를 설계하였다.
3.  **효율성 및 성능 입증**: 7개의 주요 벤치마크 데이터셋에서 실시간 속도를 유지하면서도 최첨단(SOTA) 성능을 달성하였음을 보여준다.

## 📎 Related Works

### DCF 기반 트래킹
DCF 트래커들은 순환 컨볼루션(Circular Convolution)을 주파수 영역(Frequency domain)으로 변환하여 빠르게 계산하는 특성을 가진다. 최근에는 사전 학습된 딥러닝 네트워크의 특징을 활용하여 정확도를 높이려는 시도가 있었으나, 오프라인 특징 추출기만으로는 딥 CNN의 잠재력을 완전히 활용하지 못했다는 한계가 있다.

### Siamese 네트워크 기반 트래킹
두 개의 브랜치(Template, Search region)를 통해 특징 맵의 유사도를 측정하여 타겟을 찾는다. SiamFC 이후 온라인 업데이트를 도입한 CFNet 등이 등장하며 성능이 향상되었으나, 앞서 언급한 Model Drift 문제가 여전히 존재한다. 최근에는 트래킹 문제를 One-shot detection 문제로 정의하여 대규모 데이터셋으로 학습시킨 오프라인 모델들이 경쟁력 있는 성능을 보이고 있다.

## 🛠️ Methodology

### 전체 파이프라인
전체 시스템은 ROI(Region of Interest) 패치를 입력받아 **DCF $\rightarrow$ Proposal Generation $\rightarrow$ ASN** 순으로 처리한다.
1.  **DCF 단계**: 현재 프레임에서 타겟의 대략적인 위치 $P_i$와 스케일 $C_i$를 예측한다.
2.  **Proposal 생성**: DCF의 결과값에 가우시안 섭동(Gaussian perturbation)을 더해 타겟이 존재할 가능성이 높은 영역들을 샘플링하여 여러 개의 Proposal을 생성한다.
3.  **ASN 단계**: 생성된 Proposal들을 ASN에 입력하여 가장 높은 신뢰도 점수를 가진 최종 상태 $S_i$를 결정한다.

### DCF 기반 Coarse 예측
DCF 모듈은 계산량을 줄이기 위해 단순한 구조를 가지며, 다음과 같은 손실 함수를 최소화하는 필터 $w$를 학습한다.

$$l(w;x) = \sum_{j=1}^{N} \sum_{i=1}^{D} \mu_j \| f(w, x_{ji}) - y_{ji} \|^2 + \sum_{i=1}^{D} \| \lambda f_i \|^2$$

여기서 $y_{ji}$는 원하는 응답(Desired response)이며, $\lambda$는 경계 효과(Boundary effects)를 방지하기 위한 정규화 파라미터이다. 효율적인 계산을 위해 FFT(Fast Fourier Transform)를 통해 주파수 영역으로 변환한 뒤, Conjugate Gradient(CG) 방법을 사용하여 최적의 필터를 도출한다.

### ASN 기반 Fine 정밀 위치 추정
ASN은 템플릿과 탐색 영역 간의 잠재적 관계를 캡처하며 두 개의 서브넷으로 구성된다.

-   **Attention Subnet**: 템플릿 이미지에서 ResNet-50 백본을 통해 특징을 추출하고, Channel Attention 블록과 Xception 블록을 거쳐 공간 크기를 압축(Precise ROI Pooling)한다. 최종적으로 $1 \times 1$ 컨볼루션을 통해 각 채널의 중요도를 나타내는 **채널 가중치**를 생성한다.
-   **Estimate Subnet**: 탐색 영역의 Proposal 특징을 추출한다. Attention Subnet에서 생성된 채널 가중치를 요소별 곱($\otimes$) 연산으로 적용하여 특징 맵을 정제한다. 이후 Global Average Pooling(GAP)과 FC-layer를 통해 해당 Proposal의 최종 신뢰도 점수(Confidence score)를 예측한다.

### 학습 및 손실 함수
ASN은 오프라인으로 학습되며, 예측된 점수와 실제 Ground-truth 간의 GIoU(Generalized Intersection over Union) 차이를 최소화하는 MSE(Mean Squared Error) 손실 함수를 사용한다.

$$L^{ASN}(P_{gt}, P_{mn}, Score_{mn}) = \| GIOU(P_{gt}, P_{mn}) - Score_{mn} \|^2$$

## 📊 Results

### 실험 설정
-   **데이터셋**: OTB100, VOT2018, LaSOT, GOT-10k, TrackingNet, UAV123, UAVDT.
-   **구현**: Python/PyTorch 기반, NVIDIA 1080ti GPU 사용.
-   **비교 대상**: BACF, STRCF, MCCT, SiamFC, ECO, CFNet, SiamRPN++, ATOM, DiMP 등.

### 주요 결과
-   **정량적 성능**: OTB100과 VOT2018에서 경쟁력 있는 AUC와 Precision을 달성했으며, 특히 VOT2018의 EAO(Expected Average Overlap)에서 최상위 성능을 보였다.
-   **실시간성**: 단일 GPU에서 약 **38 FPS**로 동작하여 실시간 요구사항을 충족한다.
-   **대규모 데이터셋**: LaSOT에서 DiMP에 근접하는 AUC(0.565)를 기록했으며, Precision(0.580)에서는 DiMP를 앞질렀다.
-   **공중 촬영 시나리오(Aerial Scenario)**: UAV123 및 UAVDT 데이터셋에서 매우 강건한 성능을 보였다. 특히 빠른 카메라 움직임이나 Motion Blur가 발생하는 상황에서 Coarse-to-fine 전략이 유효함을 입증하였다.
-   **모델 효율성**: DiMP와 비교했을 때 파라미터 수가 약 절반 수준(211MB vs 364MB)임에도 불구하고 유사하거나 더 나은 성능을 보여 파라미터 이용 효율이 높음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
-   **상호 보완적 구조**: DCF의 빠른 초기 추론과 ASN의 정밀한 보정이 서로를 보완한다. DCF가 검색 범위를 효과적으로 좁혀줌으로써 ASN이 더 정밀한 확인 과정을 거칠 수 있게 한다.
-   **구성 요소의 유효성**: Ablation Study를 통해 Xception 블록과 GIoU 손실 함수가 성능 향상에 필수적임을 확인하였다. 단순한 Channel Attention이 복잡한 Dual Attention보다 더 효과적이었다는 점은 트래킹 작업에서 전역적 문맥 정보보다 채널별 특징 선택이 더 중요함을 시사한다.

### 한계 및 비판적 해석
-   **백본 깊이의 한계**: ResNet-101과 같은 더 깊은 네트워크를 사용하더라도 성능 향상이 뚜렷하지 않았으며, 오히려 파라미터 수만 증가시키는 결과가 나타났다. 이는 모델의 깊이보다 특징을 어떻게 정제하고 가중치를 부여하느냐(Attention mechanism)가 더 핵심적인 요소임을 의미한다.
-   **가정**: 본 모델은 템플릿 이미지가 첫 프레임에서 정확하게 주어졌다는 가정을 전제로 하며, 이후의 업데이트는 DCF 모듈에 의존한다.

## 📌 TL;DR

본 논문은 온라인 업데이트 DCF와 오프라인 학습 ASN을 결합한 **Coarse-to-fine 비주얼 트래킹 프레임워크**를 제안한다. DCF가 타겟 위치를 대략적으로 예측하고, ASN이 채널 어텐션을 통해 이를 정밀하게 보정함으로써 **실시간 속도(38 FPS)**와 **SOTA급 정확도**를 동시에 달성하였다. 특히 모델 크기를 최적화하면서도 공중 촬영과 같은 극한 환경에서 강건한 성능을 보여, 향후 효율적인 실시간 트래커 설계에 중요한 방향성을 제시한다.