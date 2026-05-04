# Multiscale Progressive Text Prompt Network for Medical Image Segmentation

Xianjun Han, Qianqian Chen, Zhaoyang Xie, Xuejun Li, Hongyu Yang (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 정확한 형태학적 통계를 얻기 위해 필수적인 과정이다. 그러나 딥러닝 기반의 분할 모델을 학습시키기 위해서는 방대한 양의 레이블링된 데이터가 필요하며, 이는 의료 분야에서 데이터 확보의 어려움과 높은 비용이라는 문제를 야기한다. 또한, 기존의 다중 스케일(Multiscale) 정보 융합 방식들은 인코더에서 생성된 표현을 디코더에서 순차적으로 처리하기 때문에, 스케일 간의 시맨틱 갭(Semantic Gap)을 완전히 해결하지 못하고 정보 융합이 불충분한 경향이 있다.

본 논문의 목표는 텍스트 프롬프트(Text Prompt)를 사전 지식(Prior Knowledge)으로 활용하여, 적은 양의 데이터로도 고정밀 의료 영상 분할을 달성하는 '다중 스케일 점진적 텍스트 프롬프트 네트워크'를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 자연어 텍스트를 의사 레이블(Pseudolabels)처럼 활용하여 이미지 분할 과정을 가이드하는 것이다. 이를 위해 다음과 같은 설계 전략을 채택하였다.

1.  **2단계 학습 파이프라인**: 방대한 자연 이미지-텍스트 쌍을 이용한 Contrastive Learning 기반의 사전 학습(Pre-training) 단계와, 이를 기반으로 의료 영상 분할을 수행하는 다운스트림(Downstream) 단계로 구성하여 의료 데이터 부족 문제를 해결하였다.
2.  **하이브리드 PPE (Prior Prompt Encoder)**: CNN의 지역적 특징 추출 능력과 Transformer의 전역적 의존성 캡처 능력을 결합한 U-shaped Transformer 구조를 설계하여 이미지와 텍스트의 멀티모달 특징을 효율적으로 추출한다.
3.  **점진적 다중 스케일 융합 (MSFF)**: 단일 스케일의 멀티모달 특징을 다중 스케일 표현으로 확장하는 MSFF(Multiscale Feature Fusion) 블록을 도입하여, 자연 데이터와 의료 데이터 간의 시맨틱 갭을 메우고 예측 정확도를 높였다.
4.  **UpAttention 기반 정교화**: 최종 마스크 생성 전, Global Average Pooling과 Max Pooling을 포함한 UpAttention 블록을 통해 세밀한 경계선을 복원하고 결과를 정교화한다.

## 📎 Related Works

### 1. 멀티모달 의료 영상 분할
CT, MRI, PET 등 여러 모달리티를 융합하는 방식이 연구되어 왔으나, 환자 데이터의 개인정보 보호 및 지적 재산권 문제로 인해 짝지어진(Paired) 데이터셋을 찾기 어렵다는 한계가 있다. 본 논문은 이미지 모달리티 대신 계산 비용이 훨씬 저렴하고 획득이 용이한 텍스트 프롬프트를 활용함으로써 이 문제를 해결하고자 한다.

### 2. Contrastive Learning (CL)
SimCLR, MoCo 등 기존의 CL 방식은 주로 전역적 표현(Global Representation)을 학습하므로 픽셀 단위의 분할 작업에는 부적합한 면이 있다. 본 논문은 SimSiam의 전략을 차용하여 PPE를 사전 학습시키며, 특히 이미지와 텍스트의 상호 유사성을 극대화하는 방향으로 학습을 진행한다.

### 3. CNN 및 Transformer 기반 모델
CNN(예: U-Net)은 지역적 특징 추출에 강하지만 전역적 문맥 파악에 취약하고, Transformer(예: ViT)는 전역적 관계는 잘 파악하지만 지역적 세부 정보와 해상도 유지 능력이 부족하다. 본 논문은 이 두 구조를 결합한 하이브리드 형태의 PPE를 통해 두 장점을 모두 취하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
모델은 크게 두 단계로 작동한다. **Stage 1**에서는 자연 이미지-텍스트 쌍을 통해 PPE를 사전 학습시키고, **Stage 2**에서는 학습된 PPE를 의료 영상 분할 작업에 적용하여 MSFF와 UpAttention을 통해 최종 마스크를 생성한다.

### 2. Stage 1: PPE 및 Contrastive Learning
PPE는 이미지 입력 브랜치와 텍스트 입력 브랜치로 나뉜다.
- **이미지 브랜치**: ConvBN 블록과 DownBlock(MaxPooling + ConvBN)을 통해 다단계 이미지 특징 $X_1, X_2, X_3$를 추출한다.
- **텍스트 브랜치**: BERT를 통해 텍스트를 768차원의 벡터 $X_{text}$로 변환한 후, 1D Convolution 층을 거쳐 처리한다.
- **U-shaped Transformer**: 이미지 패치 임베딩 $X_{p,1}$에 텍스트 특징을 더해 Transformer 블록에 입력하며, DownViT와 UpViT를 통해 멀티모달 특징을 융합한다.

**학습 절차**:
두 개의 동일한 PPE 모델을 사용하여, 이미지의 서로 다른 증강 뷰(Augmented Views) 간의 특징 유사성을 최대화하는 Contrastive Learning을 수행한다. 이때 한쪽 모델에 stop-gradient 연산을 적용하여 모델의 붕괴(Collapse)를 막고 수렴을 돕는다.

### 3. Stage 2: MSFF 및 UpAttention
**Multiscale Feature Fusion (MSFF)**:
PPE에서 출력된 단일 스케일 특징들을 $\text{tensor}_1, \text{tensor}_2, \text{tensor}_3$라고 할 때, 이를 다음과 같은 과정을 통해 점진적 다중 스케일 특징으로 변환한다.
- **Patch Merging**: $2 \times 2$ 인접 패치를 결합하여 해상도를 낮추고 채널을 확장한다.
- **Patch Expanding**: 채널을 분해하여 해상도를 높인다.
- 위 과정을 통해 생성된 3가지 특징 그룹에서 동일 크기의 텐서들을 채널 방향으로 결합(Concatenate)하여 더 풍부한 시맨틱 정보를 가진 $\text{new\_tensor}_{1,2,3}$를 생성한다.

**UpAttention**:
CBAM(Convolutional Block Attention Module)에서 영감을 얻어, GAP(Global Average Pooling)와 GMP(Global Max Pooling)를 병렬로 처리하고, 업샘플링된 하위 스케일 특징과 결합하여 최종 마스크를 정교화한다.

### 4. 손실 함수 (Loss Function)
모델은 Weighted Binary Cross Entropy (WBCE)와 Weighted Dice Loss의 결합을 사용한다.

- **WBCE**: 픽셀 단위 분류를 위해 사용하며, 정답 레이블의 양수/음수 픽셀 비중을 조절하기 위해 가중치 $w_1, w_2$를 곱한다.
$$L_{WBCE}(x,y) = \frac{w_1 \cdot \text{pos} \cdot L_{BCE\_pos}}{\text{pos}_{sum}} + \frac{w_2 \cdot \text{neg} \cdot L_{BCE\_neg}}{\text{neg}_{sum}}$$
- **Weighted Dice Loss**: 클래스 불균형 문제를 해결하기 위해 사용하며, 라플라스 스무딩($10^{-12}$)을 적용한다.
$$L_{WDice}(x,y) = 1 - \frac{1}{N} \sum_{i=0}^{N-1} \frac{w_1 p(x_i) \cdot w_2 y_i + \text{smooth}}{w_1 p(x_i)^2 + w_2 y_i^2 + \text{smooth}}$$
- **최종 손실**: $$L = 0.5 \cdot L_{WBCE} + 0.5 \cdot L_{WDice}$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 사전 학습에는 COCO 데이터셋을 사용하였으며, 평가에는 의료 데이터셋(MoNuSeg, QaTa-COV19, Glas)과 자연 이미지 데이터셋(CoSKEL, MFFD)을 사용하였다.
- **텍스트 어노테이션**: 의료 데이터셋은 수작업 또는 기존 연구(LViT)의 패턴을 따랐으며, 자연 이미지 데이터셋은 BLIP 모델을 통해 자동으로 생성하였다.
- **비교 대상**: MedT, GTUNet, SwinUNet, UCTransNet, LViT.
- **평가 지표**: Dice, mIoU, Accuracy (Acc), Precision, Recall.

### 2. 정량적 결과
표 1과 표 2에 따르면, 제안 모델은 모든 데이터셋에서 비교 대상 모델들보다 우수한 성능을 보였다.
- **의료 데이터**: MoNuSeg에서 Dice 80.59%, QaTa-COV19에서 Dice 91.53%, Glas에서 Dice 88.12%를 달성하여 SOTA(State-of-the-art) 수준의 성능을 기록하였다.
- **자연 이미지**: CoSKEL(Dice 79.32%)과 MFFD(Dice 79.92%)에서도 기존 모델들을 크게 상회하는 결과를 보였다. 특히 텍스트 기반 가이드를 사용하는 LViT보다 월등한 성능을 보였는데, 이는 PPE의 사전 학습 단계가 유효했음을 시사한다.

### 3. 정성적 및 통계적 분석
- **시각적 결과**: 제안 모델은 특히 병변의 경계선(Edge) 묘사에서 매우 정밀한 결과를 보여주었으며, 다른 모델들이 겪는 과잉 분할(Over-segmentation)이나 빈 공간(Hollows) 발생 문제가 적었다.
- **T-Test 분석**: ANOVA 및 T-test 분석 결과, Dice 및 IoU 점수에서 $P < 0.05$ 수준의 유의미한 성능 향상이 입증되었다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 연구의 가장 큰 강점은 **'자연 데이터 $\rightarrow$ 의료 데이터'**로 이어지는 지식 전이(Knowledge Transfer) 구조를 구축한 점이다. 의료 데이터의 희소성 문제를 해결하기 위해 대규모 자연 이미지-텍스트 쌍으로 PPE를 먼저 학습시키고, 이를 의료 영상에 적용함으로써 모델이 일반적인 시각-언어 표현 능력을 갖추게 하였다. 또한, 단순한 융합이 아닌 '점진적(Progressive)' 다중 스케일 융합(MSFF)을 통해 도메인 간의 시맨틱 갭을 효과적으로 극복하였다.

### 한계 및 논의사항
논문에서는 텍스트 프롬프트의 품질이 결과에 큰 영향을 미칠 수 있음을 시사한다. 자연 이미지의 경우 BLIP을 통해 자동 생성하였으나, 의료 영상의 경우 여전히 수작업이나 특정 패턴에 의존하고 있다. 향후 연구에서는 의료 전문 용어에 특화된 LLM을 통해 더 정교한 자동 프롬프트를 생성하는 방향으로 발전시킬 필요가 있다. 또한, 3D 의료 영상으로의 확장 가능성에 대한 구체적인 언급이 부족하여, 2D 기반의 제안 방식이 3D 볼륨 데이터에서도 동일한 효율성을 보일지는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 의료 영상 분할의 데이터 부족 문제를 해결하기 위해 **텍스트 프롬프트를 가이드로 사용하는 2단계 학습 네트워크**를 제안한다. 자연 이미지-텍스트 쌍으로 사전 학습된 **PPE**가 전역/지역 특징을 추출하고, **MSFF**가 이를 점진적인 다중 스케일 특징으로 확장하며, **UpAttention**이 최종 마스크를 정교화한다. 실험 결과, 의료 및 자연 이미지 모두에서 기존 SOTA 모델들을 능가하는 정밀한 경계 추출 성능을 보였으며, 이는 텍스트 기반 사전 지식과 다중 스케일 융합 전략이 유효함을 입증한다.