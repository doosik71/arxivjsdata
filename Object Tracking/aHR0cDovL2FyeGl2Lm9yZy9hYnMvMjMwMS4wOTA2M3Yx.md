# DASTSiam: Spatio-Temporal Fusion and Discriminative Augmentation for Improved Siamese Tracking

Yucheng Huang, Eksan Firkat, Ziwang Xiao, Jihong Zhu, Askar Hamdulla (2023)

## 🧩 Problem to Solve

본 논문은 Siamese 기반 객체 추적기(Siamese Tracker)가 직면한 두 가지 핵심적인 문제를 해결하고자 한다.

첫째는 **대상 객체의 외형 변화(Appearance Change)** 문제이다. 추적 과정에서 객체의 가로세로 비율(Aspect Ratio) 변화, 가려짐(Occlusion), 크기 변화(Scale Variation) 등이 발생하면 기존 추적기의 강건성(Robustness)이 크게 저하된다. 기존의 선형 업데이트(Linear Updating) 방식으로는 복잡한 환경에서의 적응적 타겟 기술자(Adaptive Target Descriptor)를 학습하는 데 한계가 있다.

둘째는 **배경 clutter(Background Clutter)** 문제이다. 복잡한 배경 내에 타겟과 유사한 외형을 가진 객체가 존재할 경우, 응답 맵(Response Map)에 여러 개의 높은 응답 지점이 생성되어 타겟의 위치를 정확하게 특정하지 못하고 추적에 실패하는 경우가 발생한다. 이는 타겟과 배경을 구분할 수 있는 깊은 수준의 시맨틱 정보(Depth Semantic Information)가 부족하기 때문이다.

따라서 본 논문의 목표는 시공간 정보(Spatio-temporal information)를 효과적으로 활용하고 타겟의 변별력을 높여, 외형 변화와 복잡한 배경 상황에서도 안정적으로 객체를 추적할 수 있는 **DASTSiam** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 Self-attention 및 Cross-attention 메커니즘을 도입하여 Siamese 네트워크의 인코더와 디코더를 강화하는 것이다.

1.  **Spatio-Temporal (ST) Fusion Module**: 고립된 비디오 프레임 간의 관계를 연결하여 풍부한 시간적 단서(Temporal Cues)를 통합한다. 이를 통해 객체의 변형이나 가려짐과 같은 외형 변화에 대한 강건성을 높인다.
2.  **Discriminative Augmentation (DA) Module**: 템플릿과 검색 영역(Search Region) 간의 시맨틱 유사성을 분석하여 검색 영역의 특징을 강화한다. 이를 통해 배경 clutter 속에서도 타겟을 명확히 구분하는 변별력을 향상시킨다.
3.  **Label Assignment 수정**: 기존의 IoU 기반 앵커 레이블 할당 방식을 중심 거리(Center Distance) 기반 방식으로 변경하여 객체 위치 추정의 신뢰도를 높였다.

## 📎 Related Works

Siamese 기반 추적기는 일반적으로 템플릿 이미지와 검색 영역 이미지를 입력받아 상관관계 함수(Correlation Function)를 통해 응답 맵을 생성한다.

- **SiamFC 및 SiamRPN**: 초기 템플릿을 업데이트하지 않거나 단순한 선형 업데이트를 사용하므로, 외형 변화가 심한 환경에서 추적 실패율이 높다.
- **DaSiamRPN 및 RASNet**: 배경 간섭을 줄이기 위해 부정 샘플(Negative Sample) 세트를 구축하거나 특정 채널에 가중치를 두는 방식을 사용했다. 그러나 이러한 방식은 추가적인 샘플 구축 과정이 필요하며, 깊은 특징 공간(Depth Feature Space)에서의 시맨틱 차이를 충분히 활용하지 못한다는 한계가 있다.

본 논문은 이러한 한계를 극복하기 위해 Transformer 구조를 도입하여 추가적인 샘플 구축 없이도 시공간적 맥락과 시맨틱 유사성을 학습하도록 설계되었다.

## 🛠️ Methodology

DASTSiam의 전체 구조는 기존 Siamese 네트워크에 ST fusion 모듈(인코더 수정)과 DA 모듈(디코더 수정)을 추가한 형태이다.

### 1. Spatio-Temporal (ST) Fusion Module
ST 모듈은 상관관계 연산 전, 템플릿 특징 맵 $f_z$에 시간적 정보를 통합한다. 3개의 프레임 $T_i$(초기 프레임), $T_a$(이전 프레임), $T_c$(현재 프레임)에서 추출한 특징 $f_i, f_a, f_c$를 사용하며, 다음과 같이 정의된다.

$$ST(f_i, f_a, f_c) = \Phi_{ST}(\text{Encoder}(f_a, f_c)) + f_i$$

여기서 Encoder는 Cross-attention 메커니즘을 사용하여 $f_c$를 Query($Q_c$) 및 Value($V_c$)로, $f_a$를 Key($K_a$)로 사용하여 융합한다.

$$\text{CrossAttn}(Q, K, V) = \text{Softmax}\left(\frac{Q_c K_a^T}{\sqrt{d}}\right)V_c$$

또한, 연속된 프레임 간의 상태 변화가 너무 커서 발생하는 오정보(Prior Misleading)를 수정하기 위해 컨볼루션 레이어 기반의 적응형 필터 $\Phi_{MF}$를 적용하며, 최종적으로 초기 프레임의 특징 $f_i$를 더해 실제 타겟 정보가 소실되지 않도록 보장한다.

### 2. Discriminative Augmentation (DA) Module
DA 모듈은 응답 맵 생성 전, 검색 영역 특징 $f_s$의 변별력을 높이기 위해 다음과 같이 정의된다.

$$DA(f^*_z, f_s) = \Phi_{DA}(\text{Decoder}(f^*_z, f_s)) + f_s$$

이 과정은 다음과 같은 단계로 이루어진다.
1. **Self-attention**: 검색 영역 내의 유사한 특징 지점들이 서로 더 주목하게 하여 내부 시맨틱 유사성을 강화한다.
2. **Cross-attention**: 강화된 템플릿 특징 $f^*_z$와 검색 영역 특징 $f_s$를 통합하여 타겟과 배경을 구분하는 변별력 있는 마스크(Discriminative Mask)를 생성한다.
3. **Feature Enhancement**: 생성된 마스크에 $\Phi_{DA}$ 필터를 적용해 간섭 정보를 억제한 후, 이를 $f_s$와 요소별 덧셈(Element-wise addition)하여 최종 특징 $f^*_s$를 얻는다.

### 3. Training and Inference
- **Loss Function**: 회귀(Regression)에는 Smooth-L1 Loss를 사용하며, 분류(Classification)에는 Cross-Entropy Loss와 Binary Cross Entropy(BCE) Loss를 혼합하여 사용한다.
  $$L_{total} = \lambda_1 L_{cls} + L_{reg}$$
- **Label Assignment**: IoU 기반 방식 대신, 앵커 중심과 ground-truth 중심 간의 거리 기반 방식을 채택한다. 앵커의 중심 좌표 $(row_i, col_i)$와 타겟 중심 좌표 간의 거리 $\text{distance}_{x_i, y_i}$가 임계값보다 작으면 양성 샘플로 지정한다.
  $$\text{distance}_{x_i, y_i} = \sqrt{(c_{y_{lt}} + \frac{c_{y_{rb}} - c_{y_{lt}}}{2} - row_i)^2 + (c_{x_{lt}} + \frac{c_{x_{rb}} - c_{x_{lt}}}{2} - col_i)^2}$$

## 📊 Results

### 실험 설정
- **Base Trackers**: SiamFC, SiamRPN (ResNet50 백본으로 수정)
- **데이터셋**: OTB100, LaSOT, GOT-10k, VOT2018
- **지표**: Success rate, Precision, AO (Average Overlap), SR (Success Rate)

### 주요 결과
1. **정량적 성능**: OTB100 데이터셋에서 SiamRPN의 Success rate가 $64.10\%$에서 $67.09\%$로 향상되었다. 또한 VOT2018의 EAO 및 GOT-10k의 SR 지표에서 SiamRPN++ 등 최신 알고리즘보다 우수한 성능을 보였다.
2. **속성별 분석**: LaSOT 데이터셋을 통한 분석 결과, 특히 변형(Deformation), 가로세로 비율 변화(Aspect Ratio Change), 크기 변화(Scale Variation) 상황에서 ST 모듈의 효과가 뚜렷했으며, 배경 clutter 상황에서는 DA 모듈이 성능 향상에 기여했음이 확인되었다.
3. **Ablation Study**:
   - **Label Assignment**: IoU 방식보다 중심 거리 방식이 OTB100 및 GOT-10k에서 더 높은 성능을 보였다.
   - **ST/DA 모듈**: ST 모듈 단독 추가 시보다 DA 모듈을 함께 추가했을 때 성능 향상 폭이 훨씬 컸으며, 두 모듈의 결합이 DASTSiam의 최종 성능을 견인함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 Transformer의 Attention 메커니즘을 Siamese 추적기에 효율적으로 통합하여, 기존의 단순한 템플릿 업데이트 방식이 가진 한계를 극복하였다. 특히, 단순히 특징을 합치는 것이 아니라 Cross-attention을 통해 시간적 맥락을 파악하고, 변별력 있는 마스크를 생성해 배경 간섭을 억제한 점이 유효했다.

**강점**으로는 타겟의 급격한 외형 변화와 복잡한 배경이라는 두 가지 고질적인 문제를 동시에 해결하려 했으며, 이를 모듈화하여 기존의 다양한 Siamese 추적기에 이식 가능하도록 설계했다는 점이다.

**한계 및 논의사항**으로는, 논문에서 언급되었듯 연속적으로 매우 극단적인 환경(Extreme conditions)이 발생할 경우 ST 모듈의 출력이 신뢰할 수 없는 정보를 포함하게 될 가능성이 있다. 또한, Transformer 모듈의 추가로 인해 연산량이 증가했을 가능성이 있으나, 이에 대한 구체적인 추론 속도(FPS) 비교 수치는 텍스트 상에 명시되지 않아 확인이 어렵다.

## 📌 TL;DR

DASTSiam은 Transformer 기반의 **Spatio-Temporal (ST) Fusion 모듈**과 **Discriminative Augmentation (DA) 모듈**을 통해 Siamese 추적기의 강건성과 변별력을 높인 연구이다. ST 모듈은 시간적 맥락을 통합해 외형 변화에 대응하고, DA 모듈은 시맨틱 유사성을 이용해 배경 간섭을 억제한다. 여기에 중심 거리 기반의 레이블 할당 방식을 더해, OTB100, GOT-10k 등 주요 벤치마크에서 SOTA 수준의 성능을 달성하였다. 이 연구는 향후 적응적 특징 학습 및 복잡한 실세계 환경의 객체 추적 연구에 중요한 기여를 할 것으로 보인다.