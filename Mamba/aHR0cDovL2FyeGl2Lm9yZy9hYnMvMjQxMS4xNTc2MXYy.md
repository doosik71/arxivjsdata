# MambaTrack: Exploiting Dual-Enhancement for Night UAV Tracking

Chunhui Zhang, Li Liu, Hao Wen, Xi Zhou, Yanfeng Wang (2025)

## 🧩 Problem to Solve

본 논문은 야간 환경에서의 무인 항공기(UAV) 객체 추적 시 발생하는 성능 저하 문제를 해결하고자 한다. 야간 UAV 추적의 핵심적인 어려움은 낮은 조도(poor illumination)로 인해 이미지의 밝기와 대비가 낮아져, 기존의 주간 최적화 추적 알고리즘들이 제대로 작동하지 않는다는 점이다.

기존의 해결 방식은 크게 두 가지 방향으로 나뉜다. 첫째는 저조도 이미지 향상(low-light enhancement) 기법을 사용하는 것이나, 이는 지역적 세부 정보(local details)를 손실하거나 계산 비용이 매우 높다는 단점이 있다. 둘째는 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 방식이지만, 이는 주간(소스 도메인)과 야간(타겟 도메인) 간의 데이터 분포 불균형 문제와 야간 데이터셋의 부족이라는 한계가 존재한다. 따라서 본 논문의 목표는 계산 효율성을 유지하면서도 야간의 열악한 시각 정보를 보완할 수 있는 효율적인 추적 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Mamba(State Space Model)** 아키텍처를 활용한 '이중 향상(Dual-Enhancement)' 전략을 도입하는 것이다.

1.  **Mamba 기반 저조도 향상기(MLLE):** Retinex 이론을 바탕으로 조도 추정기(Illumination Estimator)와 손상 복구기(Damage Restorer)를 구성하여, 이미지의 전역적인 밝기를 개선함과 동시에 세부 구조와 디테일을 보존한다.
2.  **교차 모달 Mamba 네트워크(CMM):** 시각 정보의 부족함을 보완하기 위해 텍스트 기반의 언어 프롬프트를 도입하고, 이를 시각 정보와 효율적으로 융합하는 교차 모달 학습 체계를 구축하여 시맨틱 수준의 향상을 꾀한다.
3.  **효율성 극대화:** Mamba 모델의 선형 계산 복잡도를 활용하여, 기존 Transformer 기반 추적기 대비 추론 속도와 메모리 효율성을 획기적으로 높였다.
4.  **데이터셋 확장:** 기존 야간 UAV 추적 데이터셋에 언어 프롬프트를 직접 추가하여, 새로운 비전-언어(Vision-Language) 야간 UAV 추적 태스크를 정의하였다.

## 📎 Related Works

논문은 야간 UAV 추적을 위해 기존에 시도된 저조도 향상 기법과 UDA 기반 접근 방식을 언급한다. ADTrack과 같은 초기 연구들은 톤 매핑(tone mapping) 알고리즘을 사용하였으며, 이후 Spatial-Channel Transformer 등을 통해 조명 영향을 제거하려 했다. 그러나 이러한 방법들은 계산 부담이 크거나 이미지의 국부적 세부 사항을 유지하는 데 어려움이 있었다.

UDA 기반 방법들은 라벨이 없는 야간 데이터를 통해 학습하려 했으나, 주간 데이터와의 분포 차이가 크고 학습에 사용할 수 있는 야간 데이터 자체가 제한적이라는 한계가 명시되었다. 본 연구는 이러한 시각적 정보의 한계를 극복하기 위해 Mamba 기반의 효율적인 이미지 복구와 언어 모달리티의 추가라는 차별화된 접근 방식을 취한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
MambaTrack은 크게 **시각 브랜치(Visual Branch)**, **언어 브랜치(Language Branch)**, 그리고 이를 융합하는 **교차 모달 Mamba 네트워크(CMM)**와 최종 예측을 수행하는 **추적 헤드(Tracking Head)**로 구성된다.

### 2. Mamba 기반 저조도 향상기 (MLLE)
MLLE는 Retinex 이론인 $I = R \otimes L$ (이미지 = 반사도 $\otimes$ 조도)에 기반한다. 실제 저조도 환경의 노이즈를 모델링하기 위해 다음과 같은 방정식을 정의한다.
$$I = (R + \hat{R}) \otimes (L + \hat{L})$$
여기서 $\hat{R}$과 $\hat{L}$은 각각 반사도와 조도의 섭동(perturbation)을 의미한다. 이를 바탕으로 밝기를 높인 이미지 $I^{lu}$를 다음과 같이 계산한다.
$$I^{lu} = I \otimes \bar{L} = R + C$$
여기서 $\bar{L}$은 light-up map이며, $C$는 전체 손상 항(corruption term)이다. 최종 향상된 이미지 $I^{en}$은 다음과 같은 절차로 도출된다.
$$(I^{lu}, F^{lu}) = IE(I, L^p), \quad I^{en} = I^{lu} + DR(I^{lu}, F^{lu})$$
- **Illumination Estimator (IE):** 저조도 이미지 $I$와 조도 사전 정보 $L^p$를 입력받아 $I^{lu}$와 특징 맵 $F^{lu}$를 생성한다.
- **Damage Restorer (DR):** 인코더-디코더 구조를 가지며, IFSSM(Illumination Fusion State Space Model)을 통해 노이즈와 아티팩트를 제거하여 최종 이미지를 복원한다.

### 3. 교차 모달 Mamba 네트워크 (CMM)
CMM은 시각 임베딩 $H_x, H_z$와 언어 임베딩 $H_t$를 공유 공간으로 매핑하여 융합한다.
먼저, 언어 정보를 검색 영역 임베딩에 주입한다.
$$\bar{H}_x = H_x \otimes H_t$$
이후 템플릿과 검색 영역을 결합하여 $H_{vl} = [\bar{H}_x; H_z]$를 형성한다. 이 데이터는 정규화 층과 선형 투영을 거쳐 $h_m$이 되며, 1D 컨볼루션, SiLU 활성화 함수, 그리고 SSM(State Space Model)을 통해 다음과 같이 처리된다.
$$y_m = \text{SSM}(\text{SiLU}(\text{Conv}(f_m)))$$
최종적으로 게이팅 메커니즘을 통해 정제된 특징 $z_m$을 얻는다.
$$z_m = y_m \otimes \text{SiLU}(f_v)$$
이렇게 생성된 특징들은 다시 결합되어 언어-향상된 검색 및 템플릿 임베딩 $\tilde{H}_{vl}$이 된다.

### 4. 추적 헤드 및 손실 함수
추적 헤드는 분류(classification)와 바운딩 박스 회귀(regression)로 구성된다. 학습을 위해 다음과 같은 멀티태스크 손실 함수를 사용한다.
$$L = \lambda_1 L_1 + \lambda_{\text{GIoU}} L_{\text{GIoU}} + \lambda_{\text{focal}} L_{\text{focal}}$$
여기서 $L_1$은 좌표 오차, $L_{\text{GIoU}}$는 박스 겹침 정도, $L_{\text{focal}}$은 클래스 불균형을 해결하기 위한 손실 함수이다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** NAT2021, NAT2021L, DarkTrack2021, UAVDark70, UAVDark135 등 5개의 야간 UAV 추적 벤치마크를 사용하였다.
- **평가 지표:** AUC(Area Under Curve)와 mACC(mean Average Counting Accuracy)를 주요 지표로 사용하였다.
- **비교 대상:** STARK50, MixFormerV2, JointNLT, VLT_TT, CiteTracker 등 최신 SOTA 추적기들과 비교하였다.

### 2. 주요 결과
- **추적 성능:** MambaTrack은 5개의 모든 벤치마크에서 가장 높은 AUC 및 mACC 점수를 기록하였다. 특히 UAVDark135 데이터셋에서 $\text{mACC}$ 점수가 57.9%로 가장 높게 나타났다.
- **효율성:** 계산 효율성 면에서 압도적인 성능을 보였다. CiteTracker 대비 **추론 속도는 2.8배 빠르며**, **GPU 메모리 사용량은 50.2% 절감**하였다.
- **속도:** 실시간 추론 속도인 42 FPS를 달성하였다.
- **어트리뷰트 분석:** 빠른 움직임(FM), 조명 변화(IV), 저해상도(LR), 가려짐(OCC), 시점 변화(VC) 등 까다로운 시나리오 모두에서 기존 visual-based 및 VL-based 모델보다 우수한 성능을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석
본 연구의 가장 큰 강점은 Mamba 아키텍처의 선형 복잡도를 이용하여 고성능의 야간 추적기를 구현하면서도 실시간성을 확보했다는 점이다. 특히 MLLE 모듈은 단순한 밝기 증가가 아니라 Retinex 이론과 SSM을 결합하여 디테일을 보존하며 향상시켰기에, 추적 헤드가 타겟을 정밀하게 로컬라이즈하는 데 기여한 것으로 분석된다. 또한, 부족한 야간 데이터를 보완하기 위해 언어 프롬프트를 도입한 CMM 전략은 시각 정보가 극도로 제한된 상황에서 강력한 보조 수단이 됨을 입증하였다.

### 2. 한계 및 논의사항
논문에서는 언어 프롬프트를 수동으로 518개 생성하여 사용하였다. 이는 데이터셋 규모가 작을 때는 가능하지만, 대규모 데이터셋으로 확장할 경우 언어 어노테이션 비용이 증가할 수 있다. 또한, Mamba 모델의 특성상 시퀀스 데이터 처리에 강점이 있으나, 이미지의 2D 공간적 구조를 완벽하게 대체할 수 있는지에 대한 더 깊은 분석이 필요하다.

## 📌 TL;DR

MambaTrack은 야간 UAV 추적의 고질적인 문제인 저조도 환경을 극복하기 위해 **Mamba 기반의 저조도 향상기(MLLE)**와 **교차 모달 Mamba 네트워크(CMM)**를 제안한 모델이다. 이 모델은 이미지 복원과 시맨틱 언어 정보를 동시에 활용하여 추적 정확도를 높였으며, 특히 Mamba의 효율적인 구조 덕분에 기존 SOTA 모델 대비 메모리 사용량을 절반으로 줄이고 속도는 약 2.8배 향상시키는 획기적인 효율성을 달성하였다. 이는 향후 하드웨어 자원이 제한된 실제 UAV 플랫폼에서의 야간 추적 시스템 적용에 매우 중요한 가능성을 제시한다.