# Sketchy Bounding-box Supervision for 3D Instance Segmentation

Qian Deng, Le Hui, Jin Xie, Jian Yang (2025)

## 🧩 Problem to Solve

3D Instance Segmentation은 포인트 클라우드 내 각 객체의 카테고리를 인식하고 개별 마스크를 생성하는 핵심적인 작업이다. 하지만 기존의 Fully Supervised 방법론들은 매우 정교한 포인트 수준의 주석(point-level annotations)을 요구하며, 이는 막대한 비용과 시간이 소요된다. 이를 해결하기 위해 Bounding Box supervision을 활용하는 Weakly Supervised 방법론들이 제안되었으나, 기존 연구들은 대부분 Bounding Box가 객체에 매우 정확하고 콤팩트하게 맞물려 있다는 가정을 전제로 한다.

실제 환경에서 사람이 작성한 Bounding Box는 스케일(scaling), 평행 이동(translation), 회전(rotation) 등의 오차로 인해 불완전할 가능성이 매우 높다. 본 논문은 이러한 불완전한 박스를 **Sketchy Bounding Box**라고 정의하고, 정교하지 않은 박스 주석만으로도 고성능의 3D Instance Segmentation을 달성하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 불완전한 Bounding Box로부터 정교한 포인트 수준의 Pseudo Label을 생성하는 **Adaptive Box-to-Point Pseudo Labeler**와, 이를 활용해 점진적으로 예측 성능을 높이는 **Coarse-to-Fine Instance Segmentator**를 공동 학습(Joint Learning)시키는 것이다.

특히, 두 개 이상의 박스가 겹치는 영역에 대해 포인트와 박스 간의 유사도를 학습하여 적절한 인스턴스에 할당하는 메커니즘을 도입하였으며, 전역 정보에서 지역 정보로 단계적으로 접근하는 Multi-level Attention 구조를 통해 세밀한 인스턴스 분할을 가능케 하였다.

## 📎 Related Works

3D Instance Segmentation은 크게 Grouping-based, Detection-based, Query-based 세 가지 파이프라인으로 나뉜다. 최근에는 Query-based 방법론들이 우수한 성능을 보이고 있으나, 여전히 조밀한 포인트 수준의 주석에 의존한다.

Weakly Supervised 접근 방식으로는 Sparse point annotation이나 2D 이미지 주석을 활용하는 방법들이 제안되었다. 특히 Bounding Box를 활용한 Box2Mask, GaPro, BSNet 등의 연구가 있었으나, 이들은 모두 Axis-aligned(축 정렬) 형태의 정확한 박스를 가정한다. CIP-WPIS와 같은 일부 연구가 느슨한 박스를 다루려 했으나, 2D 모달리티에 의존한다는 한계가 있다. 본 연구는 순수 3D 데이터만을 활용하며, 더 광범위한 형태의 부정확한(sketchy) 박스에 대응한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. Sketchy Bounding Box 설정

실제의 부정확한 주석을 모사하기 위해, Ground Truth(GT) 박스 $B = [B_{min}, B_{max}]$에 다음과 같은 섭동(perturbation)을 가하여 Sketchy Bounding Box를 생성한다.

- **Scaling**: $\alpha$ 비율만큼 확장/축소 $\rightarrow B^{scaled} = [B_{min} - \alpha E, B_{max} + \alpha E]$
- **Translation**: $\beta$ 만큼 평행 이동 $\rightarrow B^{translated} = [B_{min} + \beta E, B_{max} + \beta E]$
- **Rotation**: $\gamma$ 각도만큼 회전 $\rightarrow B^{rotated} = r([B_{min}, B_{max}], M_E, \gamma\pi/180)$

여기서 $E = B_{max} - B_{min}$이며, $M_E$는 박스의 중심점이다. 이러한 연산들을 조합하여 $S_1$부터 $S_4$까지 불완전함의 정도가 높아지는 네 단계의 설정으로 실험을 진행한다.

### 2. Adaptive Box-to-Point Pseudo Labeler

불완전한 박스 정보를 정교한 포인트 수준의 Pseudo Label로 변환하는 모듈이다.

- **Point Partition**: 포인트들을 박스 외부에 있는 배경 포인트, 단일 박스 내 포인트, 다중 박스 중첩 영역 포인트의 세 그룹으로 나눈다.
- **Single Box points**: 박스 내 포인트 중 배경을 걸러내기 위해 좌표 공간과 특징 공간의 유사도를 계산한다.
  $$s_{p,B} = \cos(f_p, f_B) \times e^{-|c_B - c_p|}$$
  여기서 $f$는 특징(feature), $c$는 좌표(coordinate)를 의미한다.
- **Overlapped points**: 중첩 영역의 포인트는 불확실성이 높으므로, 중첩되지 않은 '신뢰할 수 있는 포인트($B_{rel}$)'들로 먼저 학습한 MLP를 통해 할당한다.
  $$A_{p2b} = \text{MLP}(f_p, f_p - f_{B_{rel1}}, f_p - f_{B_{rel2}})$$
- **Loss**: 신뢰할 수 있는 포인트의 라벨 $Y$를 이용해 Cross-Entropy Loss $L_{pl} = L_{CE}(A, Y)$를 통해 학습한다.

### 3. Coarse-to-Fine Instance Segmentator

예측된 거친(coarse) 결과로부터 세밀한(fine) 결과를 도출하는 구조이다.

- **Coarse Segmentation**: Query 벡터들이 전체 포인트 클라우드 특징과 상호작용하여 초기 거친 마스크와 박스를 예측한다.
- **Hierarchical Refinement (Multi-level Attention Block)**:
  1. **Global**: 전체 씬 특징과 상호작용한다.
  2. **Reliable Region**: 예측된 박스 $B_{pred}$와 마스크 기반 박스 $B_{mask}$의 IoU를 기반으로 신뢰 영역 $F_{rel}$을 추출하여 학습한다.
     $$F_{rel} = \sigma(F, M \odot e^{IoU(B_{pred}, B_{mask})})$$
  3. **Core Region**: 객체의 핵심 영역만을 포함하는 Core Box $B_{core}$를 통해 $F_{un}$을 추출하여 복잡한 컨텍스트를 해결한다.
     $$F_{un} = \sigma(F, M \odot (B_{core} \cap B_{mask}))$$

### 4. 학습 절차 및 손실 함수

Pseudo Label과 예측된 인스턴스 간의 최적 매칭을 위해 Hungarian Method를 사용한 Bilateral Matching을 수행한다. 전체 손실 함수 $L$은 다음과 같다.
$$L = L_{pl} + L_{seg}$$
여기서 $L_{seg}$는 클래스 분류($L_{CE}$), 마스크 예측($L_{BCE}, L_{dice}$), 그리고 박스 및 코어-박스 회귀($L_{L1}, L_{MSE}$) 손실의 합으로 구성된다.

## 📊 Results

### 실험 설정

- **데이터셋**: ScanNetV2, S3DIS Area 5.
- **지표**: mAP, $AP_{50}, AP_{25}$.
- **비교 대상**: Fully Supervised 방법론(Mask3D, SPFormer 등) 및 Weakly Supervised 방법론(Box2Mask, GaPro, BSNet 등).

### 주요 결과

- **성능**: ScanNetV2에서 기존의 Weakly Supervised 방법론들보다 우수한 성능을 보였으며, 특히 $AP_{25}$ 지표에서 강점을 나타냈다. S3DIS에서는 Fully Supervised Baseline인 ISBNet보다 $AP_{50}$ 기준 +3.3 높게 측정되어, 불완전한 박스 정보만으로도 매우 강력한 성능을 낼 수 있음을 입증하였다.
- **Robustness**: 박스의 불완전함 정도($S_0 \to S_4$)가 증가함에 따라 성능이 소폭 하락하지만, 급격한 붕괴 없이 견고하게 유지됨을 확인하였다.
- **Pseudo Label 품질**: 정성적 분석 결과, GaPro 등의 기존 방법론보다 객체 간 경계를 더 명확히 구분하고 배경 포인트를 효과적으로 제거하는 Pseudo Label을 생성하였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 실제 데이터 수집 과정에서 발생할 수 있는 Bounding Box의 부정확성 문제를 정면으로 다루었다. 특히 Pseudo Labeler와 Segmentator를 동시에 학습시키는 전략을 통해, 불완전한 주석 $\to$ 정교한 Pseudo Label $\to$ 정교한 예측 $\to$ 더 나은 Pseudo Label로 이어지는 선순환 구조를 구축한 점이 돋보인다. 또한, 전역-지역-핵심 영역으로 이어지는 계층적 Attention 구조가 3D 객체의 기하학적 특성을 잘 포착함을 확인하였다.

### 한계 및 논의

저자들은 실험을 통해 Bounding Box가 '심각하게' 부정확한 경우(immensely inaccurate)에는 성능이 크게 저하됨을 언급하였다. 이는 현재의 유사도 기반 할당 방식이나 Multi-level Attention만으로는 극심한 노이즈를 극복하는 데 한계가 있음을 시사한다. 향후 연구에서는 더 넓은 범위의 오차를 허용할 수 있는 강건한 매칭 알고리즘이나 추가적인 기하학적 제약 조건의 도입이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 부정확한 3D Bounding Box(Sketchy Bounding Box) 주석만을 활용하여 고성능 3D Instance Segmentation을 수행하는 **Sketchy-3DIS** 프레임워크를 제안한다. 적응형 Pseudo Labeler를 통해 박스의 노이즈를 제거하고, 계층적 Attention 기반의 Coarse-to-Fine Segmentator로 정밀한 마스크를 생성한다. ScanNetV2와 S3DIS에서 SOTA 성능을 달성하였으며, 이는 향후 저비용 고효율의 3D 데이터셋 구축 및 실용적인 3D 씬 이해 연구에 크게 기여할 것으로 기대된다.
