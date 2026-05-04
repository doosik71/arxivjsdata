# Mamba meets crack segmentation

Zhili He, Yu-Hsing Wang (2024)

## 🧩 Problem to Solve

본 논문은 토목 인프라의 안전성 평가에 필수적인 균열 세그멘테이션(Crack Segmentation)의 정밀도와 효율성을 높이는 것을 목표로 한다. 기존의 균열 세그멘테이션 네트워크는 주로 합성곱 신경망(CNN)이나 트랜스포머(Transformer) 아키텍처를 사용해 왔다. 그러나 CNN은 국소적인 수용 영역(Local Receptive Field)으로 인해 균열의 전체적인 특징을 모델링하는 글로벌 모델링 능력이 부족하며, 트랜스포머는 전역적 의존성을 캡처할 수 있지만 입력 시퀀스 길이에 따라 연산 복잡도가 이차적으로 증가($O(N^2)$)하는 효율성 병목 현상이 존재한다. 이에 본 연구는 선형 복잡도와 강력한 전역 지각 능력을 동시에 갖춘 새로운 아키텍처인 Mamba를 균열 세그멘테이션 작업에 도입하여, 효율적이면서도 전역적인 특징 추출이 가능한 모델을 구축하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba를 단순 도입하는 것을 넘어, Mamba의 동작 원리를 **주의 집중 관점(Attention Perspective)**에서 재해석하고 이를 기반으로 균열 세그멘테이션에 최적화된 **CrackMamba** 모듈을 설계한 점이다. 

주요 기여 사항은 다음과 같다:
1. **CrackMamba 설계**: Mamba의 동적 파라미터 특성이 주의 집중 메커니즘(Attention Mechanism)의 본질과 유사하다는 점에 착안하여, 주의 집중 블록의 설계 원리를 적용한 새로운 Mamba 모듈인 CrackMamba를 제안하였다.
2. **성능 및 효율성 입증**: 다양한 균열 데이터셋에서 Vim, VMamba와 같은 기존 시각적 Mamba 모듈보다 CrackMamba가 일관되게 성능을 향상시키며, 동시에 파라미터 수와 연산 비용을 줄임을 증명하였다.
3. **전역 수용 영역 검증**: 이론적 분석과 유효 수용 영역(Effective Receptive Field, ERF) 시각화를 통해 Mamba가 실제로 전역적인 수용 영역을 확보함을 입증하였다.
4. **범용적 설계 가이드 제시**: Mamba 모듈을 주의 집중 메커니즘의 원리에 맞게 설계하는 방식이 균열 세그멘테이션뿐만 아니라 다른 컴퓨터 비전 작업에도 유용한 참고 자료가 될 수 있음을 제시하였다.

## 📎 Related Works

기존의 균열 세그멘테이션 연구는 주로 다음과 같은 접근 방식을 취했다:
- **CNN 기반 모델**: 국소적 특징 추출에는 능숙하지만, 균열이 이미지 전체에 길게 분포하는 특성상 장거리 의존성을 모델링하는 데 한계가 있다.
- **Transformer 기반 모델**: Self-attention을 통해 전역 문맥 정보를 캡처하여 성능을 높였으나, 높은 계산 및 공간 복잡도로 인해 효율성이 떨어진다. 이를 해결하기 위해 Linear Attention이나 FlashAttention 등이 제안되었으나, 이는 핵심 구조인 Scaled Dot-Product Self-Attention의 근본적인 혁신이라기보다 최적화에 가깝다.
- **Visual Mamba (Vim, VMamba)**: 최근 1차원 시퀀스 모델링에서 성공을 거둔 Mamba를 비전 분야에 적용한 모델들이다. Vim은 양방향 스캔(Bidirectional Scanning)을, VMamba는 4방향 교차 스캔(Cross-Scan, SS2D)을 도입하여 2차원 이미지의 공간적 문맥을 처리하려 했다. 하지만 본 논문에서는 이러한 일반적인 시각적 Mamba 블록들이 균열 세그멘테이션 작업에서는 항상 성능 향상을 보장하지 않는다는 점을 발견하였다.

## 🛠️ Methodology

### 1. Vanilla Mamba 및 시각적 Mamba의 기초
Mamba는 상태 공간 모델(State Space Models, SSMs)을 기반으로 하며, 연속 시간 시스템의 상태 방정식 $\dot{\mathbf{h}}(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}\mathbf{x}(t)$ 및 $\mathbf{y}(t) = \mathbf{C}\mathbf{h}(t)$에서 출발한다. 이를 이산화(Discretization)하여 신경망에 적용하며, 특히 입력 값에 따라 파라미터 $\mathbf{B}, \mathbf{C}, \Delta$가 동적으로 변하는 **선택 메커니즘(Selection Mechanism)**을 통해 중요한 정보는 선택하고 불필요한 정보는 필터링한다.

### 2. CrackMamba 설계
저자들은 Mamba의 입력 의존적(Input-dependent) 파라미터 특성이 주의 집중 메커니즘의 본질과 동일하다는 'Attention Perspective'를 제시한다. 이를 바탕으로 설계된 CrackMamba의 구조는 다음과 같다:
- **기반 구조**: VMamba의 vanilla VSS 블록을 기반으로 하며, Depth-wise Convolution(DW Conv)과 SS2D(2D-Selective-Scan) 모듈을 포함한다.
- **주의 집중 맵(Attention Map, AM) 도입**: 특징 맵을 Sigmoid 함수를 통해 $(0, 1)$ 범위로 정규화하여 AM을 생성한다.
- **결합 방식**: 원래의 특징 맵과 생성된 AM을 요소별 곱셈(Element-wise Multiplication)하여 중요한 영역을 강조한다.
- **구조적 특징**: CrackSeU의 특징 융합 모듈(FFM) 및 병렬 주의 집중 모듈(PAM)의 설계를 참고하여, 두 개의 브랜치와 스킵 연결(Skip Connection)을 갖는 구조로 설계되었다.

### 3. 시스템 통합 및 학습 절차
- **네트워크 구조**: 기존의 고성능 균열 세그멘테이션 모델인 CrackSeU-B를 베이스라인으로 사용하였다. Stage 2부터 Stage 5까지의 합성곱 블록을 Mamba 블록으로 대체하였으며, 채널 수를 맞추기 위해 Point-wise Convolution(PW Conv)을 추가하였다.
- **손실 함수**: 클래스 불균형 문제를 해결하기 위해 BCE(Binary Cross-Entropy) 손실과 Dice 손실을 혼합하여 사용하였다.
  $$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{BCE} + \lambda_2 \mathcal{L}_{Dice} + \lambda_3 \mathcal{L}_{BCE\_side}$$
  여기서 $\lambda_1, \lambda_2, \lambda_3$는 각각 $1, 1, 0.1$로 설정되었다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 아스팔트 및 콘크리트 도로 균열 데이터인 **Deepcrack**과 강철 구조물 균열 데이터인 **Steelcrack** 두 가지를 사용하였다.
- **평가 지표**: 성능 측정으로는 mi IoU(mean image-wise Intersection over Union)와 mi Dice(mean image-wise Dice coefficient)를 사용하였고, 효율성 측정으로는 파라미터 수(#Param)와 연산량(MACs)을 사용하였다.

### 2. 정량적 결과
- **Mamba 블록 간 비교**: Deepcrack과 Steelcrack 모두에서 CrackMamba가 Vim, vanilla VSS, VSS 블록보다 우수한 성능을 보였다. 특히 Steelcrack 데이터셋에서 CrackMamba는 baseline 대비 mi IoU가 $5.84\%$, mi Dice가 $4.38\%$ 향상되는 큰 폭의 상승을 기록하였다.
- **기존 SOTA 모델과의 비교**: CrackSeU-B + CrackMamba 조합은 대부분의 기존 모델(U-Net, DeepLabv3+, CE-Net 등)보다 뛰어난 성능을 보였으며, 최신 모델인 BGCrack보다는 아주 약간 낮았으나, 파라미터 수는 BGCrack의 $79\%$, 연산량은 $63\%$ 수준으로 매우 효율적인 '합성 최적 모델'임을 입증하였다.

### 3. 시각적 분석 (ERF)
유효 수용 영역(ERF) 시각화 결과, 일반 CrackSeU-B는 학습 후에도 수용 영역이 국소적으로 제한되어 있었으나, CrackMamba를 적용한 모델은 학습 과정에서 수용 영역이 급격히 확장되어 이미지 전체를 아우르는 **전역 지각(Global Perception)** 능력을 갖추게 됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 Mamba 아키텍처가 단순히 효율적인 것을 넘어, 어떻게 설계하느냐에 따라 특정 도메인(예: 균열 세그멘테이션)에서의 성능이 크게 갈릴 수 있음을 시사한다. 

**강점 및 해석**:
- **설계 철학의 중요성**: Vim이나 VMamba와 같은 범용 비전 모델의 블록을 그대로 가져오는 것보다, 해당 도메인에서 이미 검증된 메커니즘(Attention)을 Mamba에 이식하는 것이 훨씬 효과적임을 보여주었다.
- **효율성과 성능의 트레이드오프 해결**: 전역 모델링 능력을 갖추면서도 파라미터 수와 MACs를 줄였다는 점은 실제 인프라 점검 시스템과 같은 실시간/저사양 환경 적용 가능성을 높인다.

**한계 및 향후 과제**:
- **통합 방식의 단순성**: 본 연구에서는 기존 CrackSeU-B의 블록을 단순 교체하는 방식을 사용하였다. Mamba의 특성에 맞게 전체 네트워크 아키텍처를 최적화한다면 더 높은 성능 향상이 가능할 것이다.
- **적용 범위의 확장**: 균열 외에도 박리(Spalling), 철근 노출 등 다른 구조적 결함 탐지나 시계열 진동 데이터 분석으로 Mamba를 확장 적용할 필요가 있다.

## 📌 TL;DR

본 연구는 전역 모델링 능력이 부족한 CNN과 연산 비용이 높은 Transformer의 한계를 극복하기 위해 **Mamba**를 균열 세그멘테이션에 도입하였다. 특히 Mamba를 주의 집중 메커니즘으로 해석한 **'Attention Perspective'**를 통해 설계된 **CrackMamba** 모듈은, 기존의 Visual Mamba 모델들보다 뛰어난 성능을 보였으며 파라미터 수와 연산량을 동시에 줄이는 성과를 거두었다. 이는 Mamba 기반 비전 모델 설계 시 주의 집중 원리를 결합하는 것이 매우 효과적임을 입증하며, 향후 다양한 토목 구조물 결함 진단 AI 모델의 기초 설계 가이드라인을 제공한다.