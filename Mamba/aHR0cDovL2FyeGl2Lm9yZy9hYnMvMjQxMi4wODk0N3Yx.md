# Selective Visual Prompting in Vision Mamba

Yifeng Yao, Zichen Liu, Zhenyu Cui, Yuxin Peng, Jiahuan Zhou (2024)

## 🧩 Problem to Solve

본 논문은 Vision Mamba(Vim) 모델을 다양한 다운스트림 비전 작업에 효율적으로 적응시키기 위한 파라미터 효율적 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 문제를 다룬다.

기존의 Visual Prompting 방법들은 주로 전역 주의 집중(Global Attention) 메커니즘을 사용하는 Vision Transformer(ViT)를 기반으로 설계되었다. 그러나 Vim은 선택적 상태 공간 모델(Selective State Space Model)을 통해 토큰 단위의 순차적 압축 및 전파 특성을 가지므로, 기존의 ViT 기반 프롬프팅 방식을 그대로 적용하는 데 한계가 있다. 특히, 시퀀스 앞에 단순히 프롬프트 토큰을 추가하는 기존 방식은 Vim의 전체 시퀀스에 걸쳐 입력 게이트(Input Gate)와 망각 게이트(Forget Gate)를 효과적으로 활성화하지 못하며, 이는 결과적으로 변별력 있는 정보의 추출과 전파를 방해하는 결과를 초래한다.

따라서 본 연구의 목표는 Vim의 독특한 순차적 특성을 고려하여, 입력 데이터에 따라 적응적으로 게이트를 활성화하고 변별력 있는 정보의 전파를 촉진하는 새로운 Visual Prompting 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 입력 데이터에 의존하는 **Selective Visual Prompting (SVP)** 메커니즘을 통해 Vim의 선택적 상태 공간 특성을 극대화하는 것이다.

주요 기여 사항은 다음과 같다:

1. **입력 의존적 선택적 프롬프트**: 고정된 프롬프트 대신 가벼운 프롬프터(Prompter)를 통해 토큰별 프롬프트를 동적으로 생성함으로써, Vim의 업데이트 및 망각 게이트를 적응적으로 활성화하고 변별력 있는 특징 전파를 강화한다.
2. **Dual-path 구조 (Cross-Inner)**: Vim이 층 간 공유 정보와 층 내부의 특수 정보를 동시에 전파한다는 점에 착안하여, 공유 파라미터를 사용하는 **Cross-Prompting**과 층별 독립 파라미터를 사용하는 **Inner-Prompting**의 이중 경로 구조를 제안한다.
3. **효율적인 적응 및 성능 입증**: 모델의 대부분을 동결하고 소수의 파라미터만 학습시킴으로써 계산 비용을 줄이면서도, 다양한 벤치마크에서 기존의 SOTA 프롬프팅 방법들과 풀 파인튜닝(Full Fine-tuning) 대비 우수한 성능을 달성하였다.

## 📎 Related Works

### State Space Model (SSM)

선형 복잡도를 가진 SSM은 긴 시퀀스의 의존성을 모델링하는 데 유리하며, 특히 Mamba와 Mamba2는 데이터 의존적인 SSM 레이어를 도입하여 언어 모델의 백본으로 성공적으로 자리 잡았다. 이를 비전 작업에 적용한 Vision Mamba(Vim)는 양방향 스캔(Bidirectional Scans)을 통해 효율적인 시각적 표현 학습을 가능하게 하였다.

### Parameter-Efficient Fine-Tuning (PEFT)

거대 모델의 모든 파라미터를 튜닝하는 것은 비용이 너무 크기 때문에, 일부 파라미터만 학습시키는 Partial-based, 보조 모듈을 추가하는 Addition-based, 그리고 입력단에 학습 가능한 토큰을 추가하는 Prompt-based 방법들이 연구되었다.

### Prompt Learning

기존의 Visual Prompting은 크게 두 가지 유형으로 나뉜다. 첫째는 VPT 시리즈처럼 이미지 토큰 시퀀스에 프롬프트 토큰을 추가하는 방식이고, 둘째는 DAM-VP처럼 이미지 위에 직접 프레임 형태의 프롬프트를 덮어씌우는 방식이다. 하지만 이러한 방법들은 Vim의 순차적 압축 특성을 반영하지 못하며, 특히 깊은 층의 게이트들을 효과적으로 활성화하지 못해 Vim에 적용했을 때 성능이 저하되는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인

SVP는 입력 이미지를 패치로 나누고 임베딩한 후, 각 Mamba 레이어의 입력 단계에서 동적으로 생성된 프롬프트를 더해주는 구조를 가진다. 전체 모델의 인코더는 동결(Frozen)시키고, 새롭게 추가된 프롬프트 모듈 $M$과 분류 헤드 $H$만을 학습시킨다.

### 주요 구성 요소 및 작동 원리

#### 1. Cross-Prompting

레이어 간의 공유 정보를 캡처하여 특징의 일관성을 유지하는 것을 목표로 한다. 여러 레이어(예: 6, 8, 12개 레이어 단위)가 파라미터를 공유하는 완전 연결 층(Fully Connected Layer) 기반의 생성기 $G^C$를 사용하여 크로스 프롬프트를 생성한다.
$$p^C_i = G^C(x_i)$$

#### 2. Inner-Prompting

각 레이어의 고유한 특성을 추출하여 모델의 변별력을 높이는 것을 목표로 한다. 각 레이어마다 독립적인 생성기 $G^I$를 두며, 이는 선형 다운 레이어($L_{down}$), SiLU 활성화 함수, 선형 업 레이어($L_{up}$)로 구성된다.
$$p^I_i = G^I(x_i) = \text{SiLU}(L_{up}(L_{down}(x_i)))$$

#### 3. 프롬프트 결합 및 통합

두 경로에서 생성된 프롬프트의 중요도를 조절하기 위해 학습 가능한 동적 스케일링 팩터 $\alpha, \beta$를 사용하여 최종 프롬프트 $\bar{p}_i$를 생성하고, 이를 원본 입력 $x_i$에 더한다.
$$\bar{p}_i = \alpha \odot p^C_i + \beta \odot p^I_i$$
$$x^p_i = x_i + \bar{p}_i$$
여기서 $\odot$은 하다마르 곱(Hadamard product)을 의미한다.

### 학습 및 최적화

학습 가능한 파라미터 집합 $M = \{G^C, G^I, \alpha, \beta\}$와 분류 헤드 $H$에 대해 교차 엔트로피 손실 함수(Cross-Entropy Loss)를 최소화하는 방향으로 최적화를 진행한다.
$$\arg \min_{M,H} L_{ce}(y, y_{gt})$$

### Vim 게이트 활성화 메커니즘 (분석)

Vim의 상태 전이 방정식은 입력 $x_i$에 의존하는 $\Delta_i, B_i, C_i$ 파라미터에 의해 결정된다. SVP를 통해 입력이 $x^p_i = x_i + \bar{p}_i$로 변경되면, 다음과 같이 게이트들이 직접적으로 활성화된다.
$$h_i = \exp(S_\Delta(x_i + \bar{p}_i) \dots) \odot h_{i-1} + S_B(x_i + \bar{p}_i)(S_\Delta(x_i + \bar{p}_i) \odot (x_i + \bar{p}_i))$$
이를 통해 모델은 불필요한 정보는 버리고 변별력 있는 정보만을 선택적으로 업데이트하여 은닉 상태(Hidden State)에 저장하고 전파할 수 있게 된다.

## 📊 Results

### 실험 설정

- **데이터셋**: HTA (10개 데이터셋), VTAB-1K (19개 벤치마크).
- **비교 대상**: Full Fine-tuning, Linear Probing, VPT, DAM-VP, AutoVP, SPT.
- **백본**: Vim-Small, ViT-Small/16, ViT-Base/16.

### 주요 결과

1. **HTA 벤치마크**: SVP는 Vim-Small 백본을 사용했을 때 평균 정확도에서 SOTA를 달성하였다. 특히 기존의 VPT 대비 5.3%, DAM-VP 대비 4.1% 높은 성능을 보였다.
2. **VTAB-1K 벤치마크**: Natural, Specialized, Structured 세 그룹 모두에서 VPT 대비 각각 3.5%, 4.7%, 13.8%의 성능 향상을 기록하였다.
3. **모델 효율성**: 훨씬 더 큰 모델이자 더 많은 데이터(ImageNet-21K)로 사전 학습된 ViT-Base 기반의 프롬프팅 방법들과 비교해도 대등하거나 더 우수한 성능을 보였다.
4. **풀 파인튜닝과의 비교**: 10개 데이터셋 중 7개에서 풀 파인튜닝보다 높은 성능을 보였으며, 이는 사전 학습된 지식을 보존하면서 타겟 태스크에 적응하는 능력이 더 뛰어남(Catastrophic Forgetting 방지)을 시사한다.

### Ablation Study

- **프롬프트 위치**: 프롬프트를 시퀀스 앞, 뒤, 중간에 삽입하는 방식보다 토큰별로 생성하여 더해주는 SVP 방식이 평균 5.2% 더 높은 성능을 보였다.
- **구성 요소**: Cross-Prompting(CP)과 Inner-Prompting(IP)을 모두 사용할 때 가장 높은 성능이 나타났으며, 특히 IP가 성능 향상에 크게 기여함을 확인하였다.
- **하이퍼파라미터**: 공유 레이어 수가 적절한 수준(중간 단계)일 때 성능이 높았으며, Inner-Prompting의 은닉 차원이 증가할수록 성능이 향상되나 증가율은 점차 둔화되었다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 Vim의 핵심인 '선택적 상태 공간' 메커니즘을 정확히 공략하였다. 기존의 ViT 기반 프롬프팅이 '추가적인 정보 제공'에 집중했다면, SVP는 **'기존 메커니즘(게이트)의 활성화'**에 집중하였다. 시각화 결과, SVP는 꽃이나 음식과 같은 핵심 객체 영역의 업데이트 게이트를 강하게 활성화하고, 접시와 같은 배경 영역은 덜 활성화함으로써 모델이 변별력 있는 특징에 집중하도록 유도함을 입증하였다.

### 한계 및 논의

논문에서는 하이퍼파라미터(공유 레이어 수, 은닉 차원)에 따른 성능 변화를 제시하였으나, 이 최적값이 데이터셋마다 어떻게 달라지는지에 대한 일반적인 가이드라인은 명시되지 않았다. 또한, 계산 복잡도 측면에서 매우 가벼운 모듈을 추가했다고 주장하지만, 모든 토큰에 대해 동적으로 프롬프트를 생성하는 과정이 추론 속도(Inference Latency)에 미치는 정량적인 영향에 대한 분석은 부족하다.

## 📌 TL;DR

본 논문은 Vision Mamba(Vim)의 순차적 정보 전파 특성에 최적화된 **Selective Visual Prompting (SVP)** 방법을 제안한다. 입력 데이터에 기반하여 토큰별 프롬프트를 생성하는 Dual-path(Cross & Inner) 구조를 통해 Vim의 업데이트 및 망각 게이트를 효과적으로 제어함으로써, 매우 적은 파라미터 업데이트만으로도 풀 파인튜닝 및 기존 ViT 기반 프롬프팅 기법을 능가하는 성능을 달성하였다. 이는 향후 SSM 기반 비전 모델의 효율적인 적응 연구에 중요한 벤치마크가 될 것으로 보인다.
