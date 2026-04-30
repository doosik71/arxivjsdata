# SAIA: Split Artificial Intelligence Architecture for Mobile Healthcare Systems

Di Zhuang, Nam Nguyen, Keyu Chen, and J. Morris Chang (2020)

## 🧩 Problem to Solve

본 논문은 모바일 헬스케어 시스템에서 딥러닝(Deep Learning, DL) 모델을 배포할 때 발생하는 **컴퓨팅 자원 부족 문제와 네트워크 의존성 문제**를 해결하고자 한다.

딥러닝 기술은 매우 높은 연산량을 요구하기 때문에 자원이 제한적인 모바일 및 IoT 기기에 직접 배포하기 어렵다. 이를 해결하기 위해 일반적으로 데이터를 클라우드 서버로 전송하여 처리하는 방식을 사용하지만, 이는 다음과 같은 심각한 한계를 가진다.
- **네트워크 불안정성:** 위성 통신이 차단되거나 방해받는 '경합 환경(Contested environments)'에서는 클라우드 기반 시스템이 완전히 작동 불능 상태가 된다.
- **실시간성 및 효율성:** 모든 데이터를 서버로 전송하는 것은 네트워크 대역폭 낭비와 지연 시간을 초래한다.

따라서 본 연구의 목표는 네트워크 가용 상태에서는 클라우드의 강력한 연산 능력을 활용하고, 네트워크가 단절된 상황에서도 로컬의 경량 AI를 통해 기본 서비스를 제공할 수 있는 **분할 인공지능 아키텍처(Split Artificial Intelligence Architecture, SAIA)**를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 클라이언트(모바일/IoT 기기)와 서버(클라우드)에 각각 서로 다른 수준의 AI 모델을 배치하고, 이를 지능적으로 제어하는 **결정 유닛(Decision Unit)**을 도입하는 것이다.

1. **하이브리드 AI 구조:** 로컬에서 작동하는 경량화된 'Embedded AI'와 서버에서 작동하는 고성능 'Networked AI'를 동시에 구축하여 환경에 따라 선택적으로 사용한다.
2. **메타 정보 기반 결정 유닛(Decision Unit):** Embedded AI가 출력한 소프트 라벨(Soft labels)과 같은 메타 정보를 분석하여, 해당 샘플을 로컬에서 처리할지 아니면 서버로 전송할지를 결정하는 경량 이진 분류기를 제안한다.
3. **전송량 조절 파라미터 $\epsilon$ 도입:** 가중 손실 함수에 하이퍼파라미터 $\epsilon$을 도입하여, 네트워크 상태나 정확도 요구사항에 따라 서버로 전송되는 데이터의 양을 정밀하게 튜닝할 수 있게 하였다.

## 📎 Related Works

논문에서는 기존의 모델 경량화 및 분할 방식의 한계를 다음과 같이 지적한다.

- **Compact/Compressed DNNs:** SqueezeNet, MobileNet, 지식 증류(Knowledge Distillation) 등의 기법은 모델 크기를 줄이지만, 여전히 서버급 앙상블 모델의 성능에는 미치지 못한다. 또한, 로컬에만 배포할 경우 서버의 고성능 모델을 활용할 기회를 상실한다.
- **Split-DNN Architectures:** DNN을 Head와 Tail 섹션으로 나누어 클라이언트와 서버에 분산 배치하는 방식이다. 하지만 이러한 방식은 대개 서버와의 통신이 필수적이며, 통신이 완전히 단절된 상태에서 로컬 모델만으로 독립적인 작동을 수행하는 능력이 부족하다.

**SAIA와의 차별점:** SAIA는 단순한 모델 분할을 넘어, 기기의 상태(저장 공간, 전력, 대역폭)와 데이터 특성에 따라 AI 사용 위치를 동적으로 조정하며, 완전한 오프라인 상태에서도 작동 가능하다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조
SAIA는 **데이터 전처리 인터페이스 $\rightarrow$ Embedded AI $\rightarrow$ 결정 유닛(Decision Unit) $\rightarrow$ Networked AI** 순으로 구성된다.

### 2. 주요 구성 요소 및 역할

#### (1) 데이터 전처리 인터페이스 (Data Pre-processing Interface)
- **객체 탐지(Object Detection):** Faster R-CNN을 사용하여 복잡한 배경에서 관심 영역(ROI)을 분리한다.
- **의미론적 분할(Semantic Segmentation):** Otsu's thresholding 알고리즘을 사용하여 ROI를 효율적으로 추출한다.
- **특징 추출(Feature Extraction):** 
    - 구조적 특징: 비대칭 지수, 원형도 등 9가지 지표 추출.
    - 색상 특징: 조명 변화에 강인한 CIELUV 색 공간을 사용하여 3개 채널의 히스토그램(총 $3 \times 255$개 특징)을 생성한다.
    - 질감 특징: LBP(Local Binary Patterns) 분석을 통해 26개의 질감 특징을 추출한다.

#### (2) 클라이언트 측 Embedded AI
자원 제한적인 환경을 고려하여 경량 머신러닝 알고리즘을 사용한다. 본 논문에서는 SVM, RF, DART를 비교하였으며, 성능과 저장 공간 효율이 가장 좋은 **DART(Dropouts meet Multiple Additive Regression Trees)**를 최종 채택하였다.

#### (3) 서버 측 Networked AI
최고 수준의 정확도를 위해 12개의 고급 CNN 아키텍처(SENet154, EfficientNet-B7 등)를 결합한 **다중 분류기 융합(Multi-classifier Fusion)** 방식을 사용한다.
- **융합 방식:** 각 분류기 $M_k$가 예측한 클래스 $C_m$에 대한 사후 확률 $p_{mkj}$를 가중 합산하여 최종 결정한다.
$$P^m(j) = \sum_{i=1}^{k} w_i \cdot p_{mij}$$
본 연구에서는 모든 분류기에 동일한 가중치 $w_i = 1/k$를 부여하는 평균 융합 전략을 사용하였다.

#### (4) 분할 AI 결정 유닛 (Decision Unit)
Embedded AI의 예측 확률(Soft labels)을 메타 정보로 사용하여 샘플의 전송 여부를 결정하는 경량 이진 분류기이다. 특히, 서버 전송 필요성을 조절하기 위해 다음과 같은 가중 손실 함수를 사용한다.

$$L^{(t)} \approx \sum_{i=1}^{n} S(y_i) \cdot \left[ l(y_i, \hat{y}^{(t-1)}) \cdot g_i \cdot f_t(x_i) + \frac{1}{2} \cdot h_i \cdot f_t^2(x_i) \right] + \Omega(f_t)$$

여기서 가중치 함수 $S(y_i)$는 다음과 같이 정의된다.
$$S(y_i) = \begin{cases} \epsilon, & \text{if } y_i = 1 \text{ (전송 대상)} \\ 1, & \text{if } y_i = 0 \text{ (로컬 처리)} \end{cases}$$
- $\epsilon$은 하이퍼파라미터로, 이 값을 높이면 결정 유닛이 샘플을 서버로 보낼 확률(True Positive Rate)이 높아진다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** ISIC 2019(피부 병변 분석, 8개 클래스), Onychomycosis(손톱 무좀 분석, 2개 클래스).
- **환경:** 클라이언트는 Google Pixel 4 XL, 서버는 NVIDIA GTX 1080Ti 4장을 탑재한 고성능 서버를 사용하였다.

### 2. 정량적 결과
- **효과성(Effectiveness):** $\epsilon$ 값을 증가시킴에 따라 SAIA의 정확도는 점진적으로 상승하여 결국 Networked AI 단독 사용 시의 SOTA 성능에 수렴하였다. 
    - 피부 병변 데이터셋의 경우, $\epsilon=25$일 때 전체 데이터의 70%만 서버로 보내고도 Networked AI와 거의 동일한 정확도(90% vs 90.6%)를 달성하였다.
- **효율성(Efficiency):** 결정 유닛을 통해 불필요한 전송을 줄임으로써, 모든 데이터를 서버로 보낼 때보다 처리 시간(Elapsed Time)을 단축하였다.
    - 피부 병변 데이터셋 기준: Networked AI 단독(2.51s) $\rightarrow$ SAIA (1.89s)로 처리 시간 감소.

### 3. 결정 유닛 분석
- $\epsilon$이 증가함에 따라 True Positive Rate(TPR)와 False Positive Rate(FPR)가 모두 증가하지만, TPR의 증가 속도가 훨씬 빨라 효율적으로 서버 전송 대상을 선별함을 확인하였다.

## 🧠 Insights & Discussion

### 강점
- **유연한 적응성:** $\epsilon$ 파라미터를 통해 네트워크 대역폭 상황과 요구 정확도 사이의 트레이드-오프를 정밀하게 제어할 수 있다.
- **가용성 확보:** 통신이 완전히 두절된 환경에서도 Embedded AI를 통해 최소한의 의료 서비스를 제공할 수 있는 구조적 강점을 가진다.

### 한계 및 논의사항
- **메타 정보 의존성:** 결정 유닛의 성능이 Embedded AI가 제공하는 소프트 라벨의 품질에 크게 의존한다. 만약 Embedded AI가 완전히 잘못된 확신(Over-confidence)을 가지고 예측한다면, 결정 유닛이 이를 로컬 처리로 오판하여 정확도가 떨어질 위험이 있다.
- **가정 사항:** 본 논문은 전처리 단계에서 Faster R-CNN과 같은 모델이 모바일에서 충분히 빠르게 작동한다고 가정하고 있으나, 실제 배포 시에는 이 부분의 오버헤드에 대한 추가 분석이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 모바일 헬스케어 시스템을 위해 **로컬(Embedded AI)과 클라우드(Networked AI)를 결합한 분할 AI 아키텍처(SAIA)**를 제안한다. 핵심은 **결정 유닛(Decision Unit)**이 메타 정보를 기반으로 데이터의 처리 위치를 지능적으로 결정하며, $\epsilon$ 파라미터를 통해 전송량을 조절하는 것이다. 실험 결과, SAIA는 서버급 정확도를 유지하면서도 데이터 전송량과 처리 시간을 유의미하게 줄였으며, 특히 네트워크가 불안정한 극한 환경에서도 작동 가능한 의료 AI 시스템의 가능성을 제시하였다.