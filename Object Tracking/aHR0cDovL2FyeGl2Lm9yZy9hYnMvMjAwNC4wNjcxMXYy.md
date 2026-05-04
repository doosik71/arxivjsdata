# Deformable Siamese Attention Networks for Visual Object Tracking

Yuechen Yu, Yilei Xiong, Weilin Huang, Matthew R. Scott (2021)

## 🧩 Problem to Solve

본 논문은 시각적 객체 추적(Visual Object Tracking) 분야에서 Siamese 기반 추적기들이 가지는 근본적인 한계점을 해결하고자 한다. 기존의 Siamese 아키텍처는 타겟 템플릿(target template)을 온라인으로 업데이트하지 않으며, 템플릿과 검색 이미지(search image)의 특징을 각각 독립적으로 계산하는 구조를 가진다.

이러한 독립적 특징 추출 방식은 배경 문맥 정보(background context information)를 완전히 배제하게 만들며, 이는 결과적으로 복잡한 배경 속에서 타겟과 유사한 방해 요소(distractors)가 존재하거나, 타겟의 외형 변화, 심한 변형(deformation), 또는 가려짐(occlusion)이 발생했을 때 추적 성능이 급격히 저하되는 추적 드리프트(tracking drift) 현상을 야기한다. 따라서 본 연구의 목표는 타겟 템플릿과 검색 이미지 간의 상호 의존성을 학습하고, 객체의 기하학적 변형에 강건하게 대응할 수 있는 새로운 Siamese attention 메커니즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Deformable Siamese Attention (DSA)** 모듈과 **Region Refinement** 모듈을 통해 타겟 표현력을 강화하고 정밀도를 높이는 것이다.

첫째, **Deformable Siamese Attention (DSA)** 메커니즘을 도입하여 Self-attention과 Cross-attention을 결합하였다. Self-attention은 공간적(spatial) 및 채널(channel) 주의 집중을 통해 강력한 문맥 정보를 학습하며, Cross-attention은 템플릿과 검색 이미지 사이의 풍부한 문맥적 상호 의존성을 통합함으로써 타겟 템플릿을 적응적으로 업데이트하는 효과를 제공한다.

둘째, 객체의 다양한 기하학적 변형을 처리하기 위해 **Deformable Convolution**을 Attention 메커니즘에 적용하였다. 이를 통해 고정된 그리드가 아닌 가변적인 위치에서 샘플링함으로써 외형 변화에 더 강건한 특징을 추출한다.

셋째, **Region Refinement** 모듈을 설계하여 SiamRPN의 출력 결과를 정밀하게 보정한다. Depth-wise cross correlation과 Deformable ROI pooling을 사용하여 타겟의 바운딩 박스(Bounding Box)와 마스크(Mask)를 동시에 정교하게 예측한다.

## 📎 Related Works

기존의 추적기들은 크게 상관 필터(Correlation Filter) 기반 방식과 Siamese 네트워크 기반 방식으로 나뉜다. 상관 필터 기반 방식은 효율적이지만 표현력에 한계가 있으며, Siamese 기반 방식(SiamFC, SiamRPN, SiamRPN++ 등)은 대규모 데이터셋을 통해 오프라인으로 학습되어 빠른 속도와 높은 성능을 보여주었다.

최근 일부 연구에서는 템플릿을 온라인으로 업데이트하거나(UpdateNet, GradNet), 주의 집중 메커니즘을 도입하여(RASNet) 성능을 개선하려 시도하였다. 그러나 기존의 attention 기반 추적기들은 템플릿과 검색 이미지의 attention을 독립적으로 계산하여 Siamese 구조의 잠재력을 완전히 활용하지 못했다는 한계가 있다. 본 논문의 SiamAttn은 Self-attention과 Cross-attention을 공동으로 계산하고 Deformable 연산을 추가함으로써, 기존 방식보다 더 강력한 변별력(discriminability)과 강건성(robustness)을 확보하였다.

## 🛠️ Methodology

### 전체 시스템 구조

SiamAttn의 전체 파이프라인은 ResNet-50 백본 네트워크를 기반으로 하며, 크게 세 가지 구성 요소로 이루어진다: **Deformable Siamese Attention (DSA) 모듈**, **Siamese Region Proposal Networks (SiamRPN)**, 그리고 **Region Refinement 모듈**이다.

### 1. Deformable Siamese Attention (DSA) Module

DSA 모듈은 템플릿 특징 $Z \in \mathbb{R}^{C \times h \times w}$와 검색 이미지 특징 $X \in \mathbb{R}^{C \times H \times W}$를 입력으로 받아 변조된 특징을 출력한다.

**A. Self-Attention**
공간적(Spatial) 및 채널(Channel) 관점에서 특징을 강화한다. 공간적 Self-attention의 경우, $1 \times 1$ 컨볼루션을 통해 쿼리($Q$)와 키($K$) 특징을 생성하고 다음과 같이 어텐션 맵 $A_{ss}^s$를 계산한다.
$$A_{ss}^s = \text{softmax}_{col}(\bar{Q}^T \bar{K}) \in \mathbb{R}^{N \times N}$$
여기서 $N = H \times W$이다. 이후 밸류($V$) 특징과 곱하여 잔차 연결(residual connection)을 통해 최종 특징 $\bar{X}_{ss}$를 얻는다.
$$\bar{X}_{ss} = \alpha \bar{V} A_{ss}^s + \bar{X} \in \mathbb{R}^{C \times N}$$
채널 Self-attention 역시 유사한 방식으로 수행되며, 두 결과의 요소별 합(element-wise sum)을 통해 최종 Self-attentional 특징 $X_s$가 생성된다.

**B. Cross-Attention**
템플릿과 검색 이미지 간의 상호 정보를 교환한다. 검색 브랜치를 기준으로 템플릿 특징 $\bar{Z}$를 사용하여 다음과 같이 채널 cross-attention 맵 $A_c$를 계산한다.
$$A_c = \text{softmax}_{row}(\bar{Z} \bar{Z}^T) \in \mathbb{R}^{C \times C}$$
이 $A_c$를 검색 특징 $X$에 인코딩하여 다음과 같이 $\bar{X}_c$를 도출한다.
$$\bar{X}_c = \gamma A_c \bar{X} + \bar{X} \in \mathbb{R}^{C \times N}$$

**C. Deformable Attention**
위에서 얻은 특징들에 $3 \times 3$ Deformable Convolution을 적용한다. 이는 객체의 기하학적 변형을 모델링하여 고정된 수용 영역의 한계를 극복하고, 타겟을 더 정확하게 식별하게 한다.

### 2. Siamese Region Proposal Networks (SiamRPN)

DSA 모듈을 통과한 특징들은 세 개의 SiamRPN 블록으로 전달된다. 각 블록은 Depth-wise cross correlation을 통해 응답 맵(response map)을 생성하며, 분류 헤드(classification head)와 회귀 헤드(regression head)를 통해 초기 추적 영역(proposal)을 예측한다.

### 3. Region Refinement Module

SiamRPN이 예측한 단일 영역을 더욱 정밀하게 보정한다.

- 여러 단계의 특징 맵에 대해 Depth-wise cross correlation을 수행하고, 이를 Fusion Block에서 정렬 및 통합한다.
- **Deformable ROI pooling**을 사용하여 타겟 특징을 더 정확하게 추출한다.
- 최종적으로 바운딩 박스 회귀와 타겟 마스크(Mask) 예측을 수행하여 위치 정밀도를 극대화한다.

### 4. 학습 절차 및 손실 함수

모델은 엔드-투-엔드(end-to-end) 방식으로 학습되며, 전체 손실 함수 $L$은 다음과 같이 정의된다.
$$L = L_{rpn-cls} + \lambda_1 L_{rpn-reg} + \lambda_2 L_{refine-box} + \lambda_3 L_{refine-mask}$$

- $L_{rpn-cls}$: Negative log-likelihood loss (분류)
- $L_{rpn-reg}, L_{refine-box}$: Smooth L1 loss (박스 회귀)
- $L_{refine-mask}$: Binary cross-entropy loss (마스크 세그멘테이션)
가중치 $\lambda_1, \lambda_2, \lambda_3$는 각각 0.2, 0.2, 0.1로 설정되었다.

## 📊 Results

### 실험 설정

- **데이터셋**: OTB-2015, UAV123, VOT2016, VOT2018, LaSOT, TrackingNet 등 6개 벤치마크에서 검증하였다.
- **비교 대상**: SiamRPN++, SiamMask, DiMP-50, ATOM 등 최신 SOTA 추적기들과 비교하였다.
- **구현 상세**: ResNet-50 백본을 사용하였으며, PyTorch 기반으로 NVIDIA RTX 2080Ti에서 구현되었다.

### 정량적 결과

- **VOT2016/2018**: VOT2016에서 EAO 0.537을 기록하며 SiamRPN++(0.464) 대비 크게 향상되었다. VOT2018에서도 EAO 0.470으로 최상위 성능을 보였다.
- **OTB-2015**: Precision 0.712, AUC 0.926으로 모든 방법론 중 가장 높은 수치를 달성하였다.
- **LaSOT & TrackingNet**: 대규모 데이터셋에서도 Normalized Precision과 Success rate 면에서 DiMP-50 등 경쟁 모델을 앞섰다.
- **추론 속도**: 마스크 헤드를 제외한 일반 추적 시 45 fps, VOT 벤치마크의 회전 박스 생성 시 33 fps의 실시간 속도를 유지하였다.

### 절제 연구(Ablation Study)

- **Attention의 영향**: Self-attention과 Cross-attention을 각각 추가했을 때 EAO가 각각 +4.7%, +4.9% 증가하였으며, 둘을 모두 적용했을 때 가장 높은 +7.3%의 향상을 보였다. 특히 Cross-attention이 성능 향상에 매우 중요한 역할을 함이 밝혀졌다.
- **Deformable Layer의 영향**: Deformable Convolution과 Pooling을 제거했을 때 성능이 소폭 하락하였으며, 이는 기하학적 변형 대응 능력이 성능 향상에 기여함을 입증한다.

## 🧠 Insights & Discussion

본 논문은 Siamese 네트워크의 고질적인 문제인 '독립적 특징 추출'과 '템플릿 업데이트 부재' 문제를 Attention 메커니즘과 Deformable 연산으로 효율적으로 해결하였다. 특히 Cross-attention을 통해 템플릿과 검색 이미지 간의 상호 작용을 모델링한 것은, 명시적인 온라인 업데이트 과정 없이도 템플릿을 적응적으로 보완하는 효과를 가져왔다는 점에서 매우 영리한 설계라고 판단된다.

또한, 단순히 바운딩 박스만 예측하는 것이 아니라 Region Refinement 모듈을 통해 마스크를 함께 예측하고 이를 통해 박스를 다시 정밀화하는 구조는 추적의 정확도를 한 단계 높였다.

다만, ResNet-50이라는 무거운 백본을 사용함에도 불구하고 실시간 속도를 유지한 점은 인상적이지만, 계산 복잡도가 높은 Attention 연산이 추가되었음에도 구체적인 연산량(FLOPs)이나 메모리 점유율에 대한 상세 분석이 부족한 점은 아쉬움으로 남는다.

## 📌 TL;DR

**SiamAttn**은 Deformable Self-attention과 Cross-attention을 결합하여 타겟의 변별력을 높이고, Region Refinement 모듈로 위치 정밀도를 극대화한 Siamese 추적기이다. 템플릿과 검색 이미지의 상호 의존성을 학습함으로써 외형 변화와 복잡한 배경에 매우 강건하며, 6개 주요 벤치마크에서 SOTA 성능을 달성함과 동시에 실시간 추적 속도를 확보하였다. 이 연구는 향후 Siamese 기반 추적기가 단순한 매칭을 넘어 적응적 특징 통합 단계로 나아가야 함을 시사한다.
