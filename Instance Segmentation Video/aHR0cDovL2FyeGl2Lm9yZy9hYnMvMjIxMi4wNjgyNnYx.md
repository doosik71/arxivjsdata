# Look Before You Match: Instance Understanding Matters in Video Object Segmentation

Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Chuanxin Tang, Xiyang Dai, Yucheng Zhao, Yujia Xie, Lu Yuan, Yu-Gang Jiang (2022)

## 🧩 Problem to Solve

최근 Video Object Segmentation (VOS) 분야에서는 현재 프레임과 과거 프레임 사이의 조밀한 매칭(dense matching)을 통해 장기적인 컨텍스트를 모델링하는 메모리 기반 방법론들이 뛰어난 성과를 거두고 있다. 그러나 이러한 접근 방식들은 개별 인스턴스에 대한 이해(instance understanding) 능력이 부족하여, 물체의 움직임이나 카메라의 시점 변화로 인해 발생하는 급격한 외형 변화(appearance variations)에 매우 취약하다는 한계가 있다.

본 논문은 VOS 작업의 본질이 비디오 내의 객체 인스턴스를 식별하고 분할하는 것인 만큼, 인스턴스 이해 능력이 필수적이라고 주장한다. 따라서 본 연구의 목표는 메모리 기반의 매칭 메커니즘에 인스턴스 이해 능력을 통합하여, 외형 변화가 심한 상황에서도 강건하게 객체를 추적하고 분할할 수 있는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간이 VOS 작업을 수행하는 방식에서 영감을 얻은 것이다. 인간은 현재 프레임에서 어떤 픽셀이 어떤 인스턴스에 속하는지를 먼저 구별(distinguish)한 뒤, 그 인스턴스를 메모리 속의 타겟 객체와 매칭(match)한다.

이를 구현하기 위해 저자들은 **ISVOS**라는 2-브랜치 네트워크를 제안한다.

1. **Instance Segmentation (IS) 브랜치**: 현재 프레임의 세부적인 인스턴스 정보를 분석하여 인스턴스 식별 능력을 제공한다.
2. **Video Object Segmentation (VOS) 브랜치**: 외부 메모리 뱅크를 활용하여 시공간적 매칭을 수행한다.

특히, IS 브랜치에서 학습된 Object Queries를 VOS 브랜치의 Query Key에 주입함으로써 '인스턴스 증강 매칭(instance-augmented matching)'을 가능하게 하며, Multi-Path Fusion (MPF) 블록을 통해 고해상도 인스턴스 인식 특징을 최종 결과에 반영한다.

## 📎 Related Works

### Propagation-based VOS

인접 프레임 간의 시간적 상관관계를 이용하여 마스크를 반복적으로 전파하는 방식이다. 초기에는 테스트 시점에 모델을 미세 조정하는 온라인 학습 방식을 사용했으나, 이후 효율성을 위해 Optical Flow 기반의 오프라인 학습 방식으로 발전했다. 하지만 이러한 방법들은 폐색(occlusion)이나 드리프트(drifting)로 인한 오류 누적 문제에 취약하다.

### Matching-based VOS

현재 프레임과 참조 프레임 또는 메모리 뱅크 간의 대응 관계를 계산하여 객체를 식별한다. 장거리 컨텍스트 모델링이 가능하다는 장점이 있으나, 대부분의 기존 모델들은 단순한 시맨틱 매칭(semantic matching)에 의존한다. 이로 인해 객체의 외형이 크게 변하거나 시점이 바뀔 때 잘못된 매칭(false match)이 발생할 위험이 크다.

### Instance Segmentation

물체 검출기를 기반으로 Bounding Box를 먼저 예측하고 마스크를 추출하는 2단계 방식에서, 최근에는 DETR과 같이 Query 기반의 '집합 예측(set prediction)' 관점으로 접근하는 1단계 모델들(예: Mask2Former)이 주류를 이루고 있다. 본 논문은 이러한 Query 기반 모델을 보조적으로 사용하여 중간 특징 단계에서 인스턴스 인식 능력을 확보한다.

## 🛠️ Methodology

### 전체 시스템 구조

ISVOS는 인스턴스 세그멘테이션(IS) 브랜치와 비디오 객체 세그멘테이션(VOS) 브랜치로 구성된다. 두 브랜치는 공통의 Backbone(ResNet)을 공유하며, 학습 단계에서는 두 작업 모두를 위해 공동 학습(joint training)된다.

### 1. Instance Segmentation (IS) Branch

IS 브랜치는 현재 프레임의 인스턴스 세부 정보를 추출하며, 다음과 같은 구성 요소로 이루어진다.

- **Pixel Decoder**: Backbone 특징 $F^{res4}$를 입력받아 픽셀 임베딩 $F^{pixel} \in \mathbb{R}^{C_{\epsilon} \times H/4 \times W/4}$과 특징 피라미드 $\{P_i\}_{i=0}^2$를 생성한다.
- **Transformer Decoder**: 학습 가능한 Object Queries $q^{ins} \in \mathbb{R}^{N \times C_d}$를 사용하여 특징 피라미드의 지역 정보를 수집한다. Masked Attention을 통해 쿼리를 업데이트하며, 그 수식은 다음과 같다.
$$q^l = \text{softmax}(M^{l-1} + q^l (k^l)^T)v^l + q^{l-1}$$
여기서 $k^l, v^l$은 특징 피라미드에서 투영된 Key와 Value이며, $M^{l-1}$은 이전 레이어의 이진화된 마스크 예측값이다. 최종적으로 업데이트된 쿼리는 $\tilde{q}^{ins}$가 된다.

### 2. Video Object Segmentation (VOS) Branch

VOS 브랜치는 IS 브랜치에서 얻은 인스턴스 정보를 활용하여 메모리 뱅크와 매칭을 수행한다.

- **Enhanced Key Encoder**: $\tilde{q}^{ins}$와 $F^{res4}$를 입력으로 하여 현재 프레임의 Query Key $Q$를 생성한다. 먼저 Deformable Attention을 통해 이미지 특징 $Q_g$를 $\tilde{q}^{ins}$에 집계하여 $\tilde{q}^{vos}$를 얻는다.
$$\tilde{q}^{vos} = \text{DeformAttn}(\tilde{q}^{ins}, p, Q_g)$$
이후 $\tilde{q}^{vos}$를 다시 $Q_g$에 주입하고 컨볼루션 투영 헤드를 거쳐 인스턴스 인식 Query Key $Q \in \mathbb{R}^{C_k \times H_m \times W_m}$를 생성한다.
- **Memory Reading**: 메모리 뱅크에서 Memory Key $K$와 Value $V$를 가져와 $Q$와의 유사도를 계산하여 어피니티 행렬(Affinity Matrix) $A$를 생성한다.
$$A_{i,j} = \frac{\exp(d(K_i, Q_j))}{\sum_i \exp(d(K_i, Q_j))}$$
여기서 $d(\cdot)$는 $L2$ 거리 함수를 사용한다. 이 $A$를 이용해 Memory Value $V$를 가중 합산하여 Readout Feature $F^{mem}$을 도출한다.
- **Memory Update**: 예측된 마스크를 ResNet18 기반의 가벼운 백본에 통과시키고 $F^{res4}$와 결합하여 새로운 Memory Value $V_{cur}$를 생성하여 저장한다.

### 3. Mask Prediction

- **IS Decoder**: $\tilde{q}^{ins}$를 이용해 카테고리 확률과 마스크 임베딩을 생성하고, 이를 $F^{pixel}$과 내적하여 인스턴스 마스크 $\hat{M}^{ins}$를 예측한다. (학습 시에만 사용)
- **VOS Decoder**: $F^{mem}$과 백본 특징 $\{B_i\}$, 픽셀 디코더 특징 $\{P_i\}$를 **Multi-Path Fusion (MPF)** 블록을 통해 융합한다. MPF 블록의 연산 과정은 다음과 같다.
$$O_i = \text{MPF}(O_{i-1}, B_i, P_i)$$
여기서 $O_{i-1}$은 이전 단계의 출력(초기값은 $F^{mem}$)이며, $\{B_i\}$와 $\{P_i\}$를 컨볼루션으로 정렬한 뒤 업샘플링 및 잔차 블록(Residual Block)을 통해 융합하여 최종 마스크를 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: DAVIS 2016, DAVIS 2017, YouTube-VOS 2018/2019.
- **지표**: Jaccard index ($J$), Boundary F-score ($F$), 그리고 이 둘의 평균인 $J\&F$를 사용한다.
- **학습**: COCO 데이터셋으로 사전 학습된 Mask2Former 가중치를 IS 브랜치에 사용하였으며, 정적 이미지 데이터셋 및 BL30K를 활용해 사전 학습을 진행했다.

### 주요 결과

- **정량적 성과**: DAVIS 2016 val (92.6%), DAVIS 2017 val (87.1%), YouTube-VOS 2018/2019 val (86.3%)를 달성하여 기존 SOTA 모델들을 상회하는 성능을 보였다.
- **강건성**: 외형 변화가 극심한 경우(예: 연기 속의 객체, 모터크로스 점프 등)에도 기존 모델(XMem, STCN, RDE)보다 훨씬 정확하고 선명한 경계의 마스크를 생성하였다.
- **장기 비디오 성능**: Long-time Video 데이터셋에서도 90.0% ($J\&F$)를 기록하며, 장기 비디오 전용 모델인 AFB-URR나 XMem보다 우수한 성능을 입증했다.

### 분석 및 토론

- **구성 요소 효과**: Query Enhancement(QE)와 MPF 블록을 제거했을 때 성능이 각각 1.4~1.9% 및 0.7~0.9% 하락하여, 두 모듈이 성능 향상에 기여함을 확인했다.
- **초기화 및 공동 학습**: Mask2Former의 가중치 초기화와 VOS/IS 작업의 공동 학습을 모두 수행했을 때 최적의 성능이 나왔으며, 이는 인스턴스 인식 능력을 유지하면서 VOS 작업에 적응시키기 위함이다.
- **메모리 효율성**: 인스턴스 증강 매칭 덕분에 더 적은 양의 메모리(예: 메모리 사이즈 2)만으로도 기존 시맨틱 매칭 기반 모델(XMem)보다 높은 성능을 유지할 수 있었다.

## 🧠 Insights & Discussion

본 논문은 단순한 픽셀 단위의 매칭을 넘어, **"먼저 인스턴스를 이해하고 그다음에 매칭하라"**는 직관을 네트워크 구조로 성공적으로 구현하였다. 특히 Query 기반의 인스턴스 세그멘테이션 모델을 보조 브랜치로 활용함으로써, VOS 모델이 객체의 개별성을 더 잘 파악하게 만들었고 이는 외형 변화에 대한 강건함으로 이어졌다.

또한, COCO 데이터셋에 없는 새로운 카테고리의 객체에 대해서도 우수한 성능을 보인 점은, 본 모델이 특정 클래스를 인식하는 것이 아니라 '인스턴스를 구분하는 일반적인 능력'을 학습했음을 시사한다.

한계점으로는 IS 브랜치를 위해 추가적인 연산량이 발생한다는 점이 있을 수 있으나, 추론 시에는 IS 브랜치의 일부(마스크 예측 부분)를 생략할 수 있어 어느 정도 완화된다. 향후 연구에서는 효율적인 메모리 저장 방식과 결합하여 정확도와 속도를 동시에 잡는 방향으로 발전할 가능성이 높다.

## 📌 TL;DR

ISVOS는 VOS의 고질적인 문제인 외형 변화에 따른 오매칭 문제를 해결하기 위해 **인스턴스 세그멘테이션 브랜치를 통합한 2-브랜치 네트워크**를 제안한다. Object Queries를 통해 인스턴스 인식 정보를 Query Key에 주입함으로써 정밀한 매칭을 수행하고, MPF 블록으로 고해상도 세부 정보를 융합한다. 결과적으로 DAVIS 및 YouTube-VOS 벤치마크에서 SOTA 성능을 달성했으며, 인스턴스 이해 능력이 VOS의 강건성을 높이는 핵심 요소임을 입증했다.
