# Vision Mamba Distillation for Low-resolution Fine-grained Image Classification

Yao Chen, Jiabao Wang, Peichao Wang, Rui Zhang, and Yang Li (2024)

## 🧩 Problem to Solve

본 논문은 저해상도 세밀 분류(Low-resolution Fine-grained Image Classification, LR FGVC) 문제를 해결하고자 한다. 세밀 분류(Fine-grained Visual Classification, FGVC)는 조립 분류(coarse-grained) 카테고리 내의 세부 하위 카테고리를 구분하는 작업으로, 일반적으로 고해상도(HR) 이미지의 세부 정보를 활용하여 높은 정확도를 달성한다. 하지만 실제 환경에서 획득하는 이미지는 촬영 거리 등의 이유로 저해상도(LR)인 경우가 많으며, 이는 결정적인 세부 정보의 결여로 이어져 모델 성능을 크게 저하시킨다.

기존의 해결책은 크게 두 가지 방향으로 나뉜다. 첫째, 초해상도(Super-resolution, SR) 기반 방식은 HR 이미지의 감독 하에 세부 정보를 복원하지만, SR 서브 네트워크가 입력 이미지 크기를 키우므로 이후의 분류 네트워크에서 파라미터 수와 연산 복잡도가 기하급수적으로 증가하는 문제가 있다. 둘째, 지식 증류(Knowledge Distillation, KD) 기반 방식은 HR 이미지로 학습된 교사(Teacher) 네트워크의 지식을 LR 이미지용 학생(Student) 네트워크로 전이한다. 그러나 높은 정확도를 위해 무거운 구조를 그대로 사용하거나, 가벼운 CNN을 사용할 경우 교사 네트워크와의 성능 격차가 크게 발생하는 한계가 있다.

따라서 본 논문의 목표는 연산 효율성을 유지하면서도 고해상도 네트워크의 성능에 근접하는 효율적인 저해상도 세밀 분류 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최근 시퀀스 모델링에서 효율성이 입증된 Vision Mamba 구조를 저해상도 세밀 분류에 도입하고, 이를 지식 증류 프레임워크와 결합하는 것이다. 주요 기여 사항은 다음과 같다.

1. **ViMD (Vision Mamba Distillation) 제안**: 하이브리드 경량 Mamba 구조를 활용하여 저해상도 세밀 분류의 정확도와 효율성을 동시에 확보한 새로운 프레임워크를 제안한다.
2. **SRVM-Net 설계**: Mamba 모델링을 통해 시각적 특징 추출 능력을 강화한 경량 초해상도 Vision Mamba 분류 네트워크(Super-resolution Vision Mamba Network)를 설계하였다.
3. **다층 Mamba 지식 증류 손실(Multi-level Mamba Knowledge Distillation Loss) 설계**: 교사 네트워크인 HRVM-Net으로부터 Logits뿐만 아니라 Hidden states의 지식을 함께 전이함으로써 학생 네트워크의 성능을 극대화하는 새로운 손실 함수를 제안한다.

## 📎 Related Works

논문에서는 기존의 LR FGVC 접근 방식을 SR 기반과 KD 기반으로 구분하여 설명한다. SR 기반 방식은 이미지의 디테일을 복원하여 정확도를 높이지만, 연산 비용의 증가로 인해 모바일이나 임베디드 기기 적용이 어렵다는 한계가 있다. KD 기반 방식은 교사-학생 구조를 통해 지식을 전달하며, 최근에는 가벼운 CNN을 학생 네트워크로 사용하는 연구가 진행되었다. 하지만 이러한 CNN 기반 학생 네트워크는 여전히 HR 교사 네트워크와의 성능 차이가 크다는 점이 한계로 지적된다.

본 연구는 이러한 한계를 극복하기 위해 기존의 CNN이나 Transformer 대신, 선형 시간 복잡도를 가지면서도 강력한 표현력을 가진 Mamba 구조를 도입하여 효율성과 성능의 트레이드-오프 문제를 해결하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인

ViMD는 크게 **SRVM-Net (Student)**, **HRVM-Net (Teacher)**, 그리고 이 둘을 연결하는 **다층 Mamba 지식 증류 손실**로 구성된다. 학습 단계에서는 먼저 HRVM-Net을 고해상도 이미지로 학습시키고, 이후 이를 고정(Frozen)한 상태에서 SRVM-Net이 저해상도 이미지를 입력받아 교사의 지식을 학습하도록 한다. 테스트 단계에서는 SRVM-Net만 단독으로 사용된다.

### SRVM-Net 구조

SRVM-Net은 SR 서브 네트워크와 ViM 분류 서브 네트워크가 순차적으로 연결된 구조이다.

1. **SR 서브 네트워크**: 저해상도 이미지 $x_l \in \mathbb{R}^{C_l \times H_l \times W_l}$을 입력받아 세부 정보를 복원하여 초해상도 이미지 $x_s \in \mathbb{R}^{C_s \times H_s \times W_s}$를 생성한다. 본 논문에서는 사전 학습된 SRGAN의 Generator를 그대로 사용하여 이미지 복원 성능을 확보하였다.
2. **ViM 분류 서브 네트워크**: 생성된 $x_s$를 입력으로 하며, 다음과 같은 구성 요소를 가진다.
    * **Patches Embedding Module**: 2D 이미지를 1D 시퀀스로 변환한다. 합성곱 연산을 통해 패치 $x_a$를 생성한 후, 평탄화(Flatten) 및 전치(Transpose) 연산을 통해 $x_b \in \mathbb{R}^{Z \times D}$를 얻는다. 이후 클래스 토큰($x_{cls}$)과 위치 임베딩($x_{pos}$)을 추가하여 최종 시퀀스 $H_0^s \in \mathbb{R}^{(Z+1) \times D}$를 구성한다.
    * **N-layers Vision Mamba Encoder**: 각 층 $E_i$는 양방향 시퀀스 Mamba 구조를 기반으로 한다. 입력 $H_i^s$는 정규화 및 선형 투영을 통해 순방향 시퀀스 $P_{fw}^i$와 역방향 시퀀스 $P_{bw}^i$로 나뉜다. 이후 $\text{SiLU}$ 활성화 함수와 Mamba 연산($M_{fw}, M_{bw}$)을 거치며, 최종 출력 $H_i^s$는 다음과 같은 잔차 구조(Residual structure)로 계산된다:
        $$H_i^s = \text{Linear}(U_{fw}^{i-1} + U_{bw}^{i-1}) + H_{i-1}^s$$
        여기서 $U_{fw}^{i-1} = \sigma(H_{i-1}^s) \odot Q_{fw}^{i-1}$이며, $\odot$는 요소별 곱셈(element-wise product)을 의미한다.
    * **Classification Head**: 최종 층의 클래스 토큰 $h_{cls}$를 선형 투영하여 예측 확률인 $\text{Logit}_s$를 산출한다.

### 다층 Mamba 지식 증류 손실

SRVM-Net의 일반화 능력을 높이기 위해 Logits와 Hidden states를 모두 활용하는 $\mathcal{L}_{MKD}$를 설계하였다.
$$\mathcal{L}_{MKD} = \mathcal{L}_{LD} + \beta \mathcal{L}_{HSD}$$

1. **Logits Distillation Loss ($\mathcal{L}_{LD}$)**: 교사와 학생의 예측 확률 분포 간의 차이를 KL Divergence로 측정한다.
    $$\mathcal{L}_{LD} = \text{KL}(\text{softmax}(\text{Logit}_s / \Delta) || \text{softmax}(\text{Logit}_t / \Delta))$$
    여기서 $\Delta$는 온도를 조절하는 하이퍼파라미터이다.
2. **Hidden States Distillation Loss ($\mathcal{L}_{HSD}$)**: $N$개 층의 각 인코더 출력값 사이의 $L_2$ 거리를 최소화한다.
    $$\mathcal{L}_{HSD} = \sum_{i=1}^N ||H_i^t - H_i^s||_2^2$$

최종 손실 함수는 분류를 위한 교차 엔트로피 손실($\mathcal{L}_{CE}$)과 증류 손실의 합으로 정의된다:
$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{MKD}$$

## 📊 Results

### 실험 설정

* **데이터셋**: CUB, CAR, DOG, PET, Flower, MIT67, Action 등 7개의 공공 세밀 분류 데이터셋을 사용하였다. HR 이미지는 $224 \times 224$ 크기이며, LR 이미지는 bicubic 보간법을 통해 $56 \times 56$ 크기로 다운샘플링하여 생성하였다.
* **비교 대상**: SR 기반 방식(DRE-Net, DME-Net)과 KD 기반 방식(SRKD, JSC)을 비교군으로 설정하였다.
* **평가 지표**: Top-1 분류 정확도, 파라미터 수, FLOPs를 측정하였다.

### 주요 결과

ViMD는 7개 데이터셋 모두에서 기존 SOTA 방법들보다 높은 정확도를 달성하였다. 구체적으로 CUB 데이터셋에서 DRE-Net 대비 6.42%p, JSC(SwinIR) 대비 2.35%p 향상된 성능을 보였다. 특히 효율성 측면에서 매우 강력한 이점을 가진다. ViMD의 분류 서브 네트워크인 Vim-Tiny는 파라미터 수가 6.99M, FLOPs가 0.50G에 불과하며, 이는 VGG16의 약 0.05%, ResNet50의 28.3%, ResNet18의 60.5% 수준이다.

### 소거 연구 (Ablation Study)

1. **구성 요소 분석**: ResNet18을 사용했을 때보다 Vim-Tiny를 사용할 때 정확도가 크게 향상되었으며(CUB 기준 6.53%p 증가), 지식 증류($\mathcal{L}_{LD}, \mathcal{L}_{HSD}$)를 적용했을 때 성능이 추가로 향상됨을 확인하였다.
2. **하이퍼파라미터 분석**: $\beta \in \{1, 10, 20, 30\}$ 범위 내에서 성능 변화를 관찰한 결과, 모델이 $\beta$ 값에 대해 비교적 강건(robust)하며 $\beta=20$일 때 전반적으로 우수한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 Mamba 구조가 가진 선형 시간 복잡도와 강력한 시퀀스 모델링 능력이 저해상도 이미지의 부족한 정보를 보완하는 데 매우 효과적임을 입증하였다. 특히, 단순한 결과값(Logits)의 전이를 넘어 내부 특징 맵(Hidden states)을 직접 전이함으로써 학생 네트워크가 교사의 정교한 특징 추출 방식을 효율적으로 학습하게 한 점이 핵심적인 성공 요인으로 분석된다.

다만, SR 서브 네트워크로 사전 학습된 SRGAN의 Generator를 그대로 사용했다는 점은 성능의 일부가 외부 모델의 성능에 의존하고 있음을 시사한다. 또한, Mamba 구조의 특성상 2D 이미지를 1D 시퀀스로 변환하는 과정에서 발생하는 공간 정보 손실을 어떻게 더 완벽하게 억제할 수 있을지에 대한 추가적인 논의가 필요할 것으로 보인다. 그럼에도 불구하고, 극도로 낮은 연산 비용으로 SOTA 성능을 낸 것은 임베디드 장치와 같은 자원 제한 환경에서 LR FGVC를 구현하는 데 있어 매우 중요한 이정표가 될 것이다.

## 📌 TL;DR

본 논문은 저해상도 세밀 분류를 위해 **Vision Mamba**와 **지식 증류(KD)**를 결합한 **ViMD** 프레임워크를 제안한다. SRGAN 기반의 초해상도 복원과 Vim-Tiny 기반의 효율적인 분류 네트워크를 구축하고, Logits와 Hidden states를 동시에 전이하는 다층 증류 손실 함수를 통해 성능을 극대화하였다. 실험 결과, ViMD는 기존 SOTA 방법들보다 훨씬 적은 파라미터(6.99M)와 연산량(0.50G FLOPs)으로도 7개 데이터셋에서 최고의 정확도를 달성하였으며, 이는 자원 제한적인 환경에서의 실시간 세밀 분류 가능성을 제시한다.
