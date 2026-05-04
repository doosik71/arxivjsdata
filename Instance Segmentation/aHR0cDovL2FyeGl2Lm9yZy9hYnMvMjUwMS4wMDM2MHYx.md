# A Novel Shape Guided Transformer Network for Instance Segmentation in Remote Sensing Images

Dawen Yu and Shunping Ji (2024)

## 🧩 Problem to Solve

본 논문은 원격 탐사 이미지(Remote Sensing Images, RSIs)에서의 인스턴스 분할(Instance Segmentation) 성능을 저하시키는 두 가지 핵심 문제를 해결하고자 한다.

첫째는 역동적인 대기 상태와 이미지 특성으로 인해 객체의 정밀한 경계(Boundary)를 추출하는 것이 어렵다는 점이다. 둘째는 광범위한 공간적 영역에 흩어져 있는 관련 객체 인스턴스 간의 상호 정보, 즉 장거리 의존성(Long-range dependency)을 통합하는 능력이 부족하다는 점이다.

따라서 본 연구의 목표는 전역적 문맥 파악 능력과 지역적 세부 형상 인지 능력을 동시에 갖춘 **SGTN(Shape Guided Transformer Network)**을 제안하여, 원격 탐사 이미지 내 객체들을 인스턴스 수준에서 정확하게 추출하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 전역적 컨텍스트 모델링을 위한 새로운 인코더와 지역적 형상 세부 정보를 강화하는 가이드 모듈을 결합하는 것이다.

1.  **LSwin (Long-range correlation boosted Swin Transformer) 인코더**: 기존 Swin Transformer의 지역 윈도우 기반 self-attention이 가진 한계를 극복하기 위해, 수직(Vertical) 및 수평(Horizontal) 방향의 1D 전역 self-attention 메커니즘을 통합하였다. 이를 통해 연산 효율성을 유지하면서도 RSIs의 광범위한 지역에 걸친 전역적 인지 능력을 확보하였다.
2.  **SGM (Shape Guidance Module)**: 객체의 경계와 형상 정보를 강조하기 위해 특화된 감독 신호(전경, 에지, 코너 포인트)를 활용하는 모듈을 도입하였다. 이는 지역적인 세부 디테일을 보존하며, 최종적으로 인스턴스 마스크의 정밀도를 높이는 역할을 한다.
3.  **통합 프레임워크**: 전역적 문맥을 잡는 LSwin과 지역적 형상을 잡는 SGM을 결합하여, 상호 보완적인 특징 추출 파이프라인을 구축하였다.

## 📎 Related Works

### 기존 연구 및 한계
- **일반 인스턴스 분할**: Mask R-CNN과 같은 픽셀 기반 방식과 PolarMask, Deep Snake와 같은 윤곽선 회귀(Contour-regression) 기반 방식으로 나뉜다.
- **원격 탐사 이미지(RSI) 적용**: RSI 특유의 복잡한 배경과 객체 다양성으로 인해 HQ-ISNet, SLCMASK-Net 등이 제안되었으나, 여전히 정밀한 경계 추출과 전역적 문맥 파악에 어려움이 있다.
- **Transformer 기반 접근**: ViT는 연산 비용이 너무 높고 해상도 손실이 있으며, Swin Transformer는 윈도우 기반의 지역 attention을 사용하므로 매우 먼 거리의 픽셀 간 상관관계를 파악하는 데 한계가 있다.

### 차별점
기존 연구들이 주로 국소적인 경계 강화나 단순한 Transformer 백본 교체에 집중했다면, 본 논문은 1D 전역 attention을 통해 연산 효율성과 전역 수용야를 동시에 잡은 **LSwin**을 제안하고, 보조 출력(foreground map)을 통해 최종 마스크를 직접적으로 정제하는 **SGM** 구조를 통해 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
SGTN은 2단계(Two-stage) 프레임워크를 따른다.
- **Stage 1**: LSwin 인코더를 통해 고차원 세만틱 특징을 추출하고, CBGM(Candidate Box Generate Module)을 통해 후보 바운딩 박스를 생성한다. 동시에 SGM이 특징 맵을 강화하여 형상 정보를 보존한다.
- **Stage 2**: 생성된 후보 박스와 SGM의 특징을 이용하여 인스턴스 마스크를 예측하고 바운딩 박스를 정밀하게 조정(Refinement)한다.

### 2. LSwin (Long-range correlation boosted Swin Transformer)
LSwin은 Swin Transformer 블록과 **LRC(Long-Range Correlation)** 블록을 교차 배치하여 구성된다.

- **Swin Transformer 블록**: 기존의 W-MSA(Window MSA)와 SW-MSA(Shifted Window MSA)를 사용하여 지역적 특징을 추출한다.
- **LRC 블록**: 2D self-attention을 수직(V-MSA) 및 수평(H-MSA) 1D attention으로 분해하여 연산한다. 이는 전역적 상관관계를 명시적으로 계산하면서도 GPU 메모리 소비를 낮춘다.

**H-MSA (수평 attention)의 연산 과정**은 다음과 같다.
입력 특징 $\text{F}$에 대해 Query($\text{Q}_{\text{row}}$), Key($\text{K}_{\text{row}}$), Value($\text{V}_{\text{row}}$)를 생성하며, 한 행(row)의 출력 특징 $\text{F}_{\text{row}}$는 다음과 같이 계산된다.

$$ \text{S}_{\text{row}} = \text{Softmax}(\text{Q}_{\text{row}} \text{K}_{\text{row}}^T) $$
$$ \text{F}_{\text{row}} = \text{S}_{\text{row}} \text{V}_{\text{row}} $$

수직 방향(V-MSA) 또한 동일한 구조로 열(column) 단위 연산을 수행한다. 최종 LSwin 블록의 출력은 Swin 블록의 결과와 LRC 블록의 결과를 학습 가능한 파라미터 $\alpha, \beta$를 이용해 가중 합산하여 생성한다.

### 3. SGM (Shape Guidance Module)
SGM은 객체의 세부 형상을 정밀하게 묘사하기 위해 설계되었다.
- **구조**: ARFEM(Adaptive Receptive Field Feature Extraction Module)을 통해 얕은 층의 세부 특징을 가져와 LSwin의 인코딩 특징과 결합하고, 4개의 컨볼루션 레이어를 통해 정제한다.
- **감독 학습**: 다중 레이블 분류 헤드를 통해 **전경(Foreground), 에지(Edge), 코너 포인트(Corner point)**를 예측하도록 학습시킨다. 이때 에지와 코너 픽셀에 대해 가중치를 2~4배 높게 설정하여 경계 학습을 강제한다.
- **마스크 정제**: SGM에서 생성된 전역 전경 분류 맵 $\text{M}_c$와 예측된 인스턴스 마스크 $\text{M}_s$를 아다마르 곱(Hadamard product)으로 결합하여 최종 마스크 $\text{M}_i$를 생성한다.

$$ \text{M}_i = \text{M}_c \odot \text{M}_s $$

## 📊 Results

### 실험 설정
- **데이터셋**: WHU(건물), BITCC(건물), NWPU VHR-10(10종 다중 클래스)
- **지표**: $\text{AP}$ (Average Precision), $\text{AP}_{50}$, $\text{AP}_{75}$, 그리고 크기별 $\text{AP}_{\text{S, M, L}}$
- **비교 대상**: YOLACT, SOLO, Mask R-CNN, CenterMask, Deep Snake, DANCE, BuildMapper 등

### 주요 결과
1.  **정량적 성능**: SGTN은 세 데이터셋 모두에서 가장 높은 $\text{AP}$를 기록하였다. 특히 LSwin 백본을 사용했을 때 $\text{ResNet-50}$이나 $\text{Swin-S}$보다 성능이 뛰어났다.
    - WHU 데이터셋: LSwin 기반 SGTN이 $\text{AP } 74.6\%$ 달성.
    - BITCC 데이터셋: $\text{AP } 52.1\%$ 달성.
    - NWPU VHR-10 데이터셋: $\text{AP } 71.3\%$ 달성.
2.  **형상 묘사 능력**: $\text{AP}_{75}$ 지표에서 타 모델 대비 높은 성능을 보여, SGM이 정밀한 경계 및 형상 예측에 크게 기여했음을 입증하였다.
3.  **백본 범용성**: LSwin 인코더를 CenterMask나 DANCE 같은 타 모델에 적용했을 때도 $\text{AP}$가 상승하는 것을 확인하여, 제안한 인코더의 범용적 우수성을 증명하였다.
4.  **효율성**: 추론 속도는 ResNet-50 기반 모델들과 유사한 수준이며, Swin-S 대비 약간의 속도 저하가 있으나 성능 향상 폭이 훨씬 크다.

## 🧠 Insights & Discussion

### 강점
본 논문은 전역적 문맥(Global Context)과 지역적 세부 형상(Local Shape)이라는 두 마리 토끼를 모두 잡았다. 특히 2D attention을 1D 수직/수평 attention으로 분해한 접근 방식은 연산 효율성을 유지하면서도 전역 수용야를 확보한 매우 실용적인 전략이다. 또한, 단순한 손실 함수 추가에 그치지 않고 보조 맵을 통한 마스크 정제(Hadamard product) 과정을 도입한 점이 실제 성능 향상에 결정적인 역할을 한 것으로 보인다.

### 한계 및 논의
- **독립적 사용 불가**: 실험 결과, LRC 블록만으로 인코더를 구성했을 때는 $\text{Swin-S}$보다 성능이 크게 낮게 나타났다. 이는 LRC가 단독 특징 추출기보다는 기존의 지역적 특징 추출기(Swin)를 보완하는 '부스터' 역할에 최적화되어 있음을 시사한다.
- **오인식 사례**: NWPU VHR-10 데이터셋에서 도로와 연결된 다리, 부두에 정박한 배 등 배경과 객체의 경계가 모호하거나 형태가 특이한 경우 여전히 오검출(FP)이나 미검출(FN)이 발생하는 한계가 관찰되었다.

## 📌 TL;DR

본 논문은 원격 탐사 이미지의 장거리 의존성 문제와 정밀한 경계 추출 문제를 해결하기 위해 **SGTN**을 제안한다. 전역적 인지를 위해 1D-Attention을 결합한 **LSwin 인코더**와 정밀한 형상 보존을 위한 **SGM 모듈**을 도입하여, 다수의 공개 데이터셋에서 기존 SOTA 모델들을 능가하는 성능을 보였다. 특히 이 연구는 원격 탐사 이미지와 같이 광범위한 영역의 컨텍스트 파악이 중요한 도메인에서 Transformer 구조를 어떻게 효율적으로 최적화할 수 있는지에 대한 중요한 방향성을 제시한다.