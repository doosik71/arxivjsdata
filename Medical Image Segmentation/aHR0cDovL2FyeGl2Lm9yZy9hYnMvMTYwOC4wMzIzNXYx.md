# Gaze2Segment: A Pilot Study for Integrating Eye-Tracking Technology into Medical Image Segmentation

Naji Khosravan, Haydar Celik, Baris Turkbey, Ruida Cheng, Evan McCreedy, Matthew McAuliffe, Sandra Bednarova, Elizabeth Jones, Xinjian Chen, Peter L. Choyke, Bradford J. Wood, Ulas Bagci (2016)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 과정은 크게 두 가지 상호 보완적인 단계인 인식(Recognition, 객체의 대략적인 위치 파악)과 묘사(Delineation, 객체의 정확한 경계 정의)로 나뉜다. 일반적으로 묘사 과정은 자동화가 상대적으로 용이하지만, 인식 단계는 매우 어려워 수동 또는 반자동 방식에 의존하거나 전수 조사(Exhaustive search)와 같은 연산 비용이 높은 최적화 기법을 사용해야 한다.

본 논문의 목표는 전문 방사선 전문의의 시선(Gaze) 정보를 인식 단계의 입력값으로 활용하여, 추가적인 사용자 상호작용 없이도 자동으로 의료 영상을 분할할 수 있는 시스템인 `Gaze2Segment`를 제안하는 것이다. 즉, 인간의 생물학적 시각 주의 집중(Visual Attention) 메커니즘을 컴퓨터 비전 기술과 결합하여 인식 문제의 효율적인 해결책을 제시하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 방사선 전문의가 영상을 판독할 때 이상 징후가 있는 영역에 더 많은 시간을 할애한다는 점에 착안하여, **시선 추적 데이터(Eye-tracking data)를 인식(Recognition) 단계의 가이드로 사용하고, 컴퓨터 비전 기반의 Saliency map과 Gradient 정보를 묘사(Delineation) 단계의 시드(Seed) 생성에 사용**하는 시너지 구조를 설계한 것이다. 이를 통해 사용자가 별도의 마우스 클릭이나 상호작용을 하지 않아도 판독 행위 자체만으로 분할 작업을 수행할 수 있는 파이프라인을 구축하였다.

## 📎 Related Works

기존의 방사선 분야 시선 추적 연구는 크게 두 가지 방향으로 진행되었다. 첫째는 방사선 전문의의 시각적 탐색 패턴을 분석하거나 전문가와 비전문가의 탐색 차이를 연구하는 심리학적 관점의 연구이다. 둘째는 시선 추적을 컴퓨터와의 상호작용 도구(Interaction tool)로 사용하는 연구로, 마우스 클릭을 대체하여 분할 작업을 수행하는 방식 등이 제안되었다.

그러나 기존 연구들은 단순히 인간의 행동을 분석하거나 수동 제어 도구로 사용하는 데 그쳤으며, 생물학적 시선 정보와 컴퓨터 비전 알고리즘을 유기적으로 결합하여 완전히 자동화된 의료 영상 분석 태스크를 수행하려는 시도는 부족했다는 점에서 본 연구와 차별점을 가진다.

## 🛠️ Methodology

`Gaze2Segment` 시스템은 총 5단계의 파이프라인으로 구성된다.

### Step 1: Eye-Tracking 및 Gaze 정보 추출
`MobileEye XG` 장비를 사용하여 전문의의 시선 데이터를 실시간으로 수집한다. 안구 카메라와 장면(Scene) 카메라를 통해 동공의 방향과 모니터의 좌표를 60Hz의 속도로 기록하며, 수집된 시선 좌표 $g_v$를 실제 CT 영상의 좌표계인 $g_s$로 변환한다. 또한, `MIPAV` 소프트웨어 플러그인을 통해 스크롤, 대비 변경 등 마우스 조작 데이터를 함께 기록하여 3D CT 스캔의 슬라이스 변동 상황을 추적한다.

### Step 2: Jitter 제거 및 Gaze 안정화
시선 데이터 특유의 미세한 떨림(Jitter)을 제거하기 위해 유클리드 거리 기반의 평활화 연산자 $J$를 적용한다. 연속된 두 시선 좌표 $g_s(i)$와 $g_s(i+1)$ 사이의 거리가 임계값 $\epsilon$ 이하일 경우, 이를 동일한 주의 영역으로 간주하여 좌표를 통합한다.
$$\text{if } ||g_s(i) - g_s(i+1)|| \le \epsilon, \text{ then } g_s(i) \text{ is set to } g_s(i+1)$$
여기서 $\epsilon$은 실험적으로 $7.5\text{mm}$로 설정되었다.

### Step 3: Visual Attention Map 생성
전문의가 의심스러운 영역에 더 오래 머문다는 가설을 바탕으로, 체류 시간(Timestamp)을 활용해 주의 집중 지도(Attention Map)를 생성한다. 각 시선 지점 $g_s(i)$에 대해 다음과 같은 Piece-wise linear 함수를 통해 주의 값 $a(i) \in [0, 1]$를 할당한다.
$$a(i) = \begin{cases} \frac{t(i) - \hat{t}}{t_{max} - \hat{t}}, & t(i) > \hat{t} \\ 0, & \text{otherwise} \end{cases}$$
여기서 $t_{max}$는 최대 체류 시간이며, $\hat{t}$는 매우 짧은 체류 시간을 가진 노이즈를 제거하기 위한 임계값이다.

### Step 4: Foreground/Background Cues 추출을 위한 Local Saliency 계산
단순한 시선 정보만으로는 정확한 경계를 알 수 없으므로, 컴퓨터 비전의 Context-aware Saliency 정의를 도입하여 객체의 특성을 파악한다.

1.  **Local Low-level**: 픽셀 $u$를 중심으로 한 패치 $p_u$와 주변 패치 $p_v$ 간의 강도 차이와 거리의 비율을 통해 Saliency를 계산한다.
    $$d(p_u, p_v) = d_{intensity} / (1 + \lambda d_{position})$$
2.  **Global Consideration**: Scale-space 접근법을 사용하여 배경과 같이 빈번하게 나타나는 특징을 억제하고, $M$개의 스케일에서 평균 Saliency $\bar{S}_u$를 구한다.
3.  **Visual Organization**: Gestalt 법칙을 적용하여 주의 집중 중심점(Foci of attention)과의 거리에 따라 Saliency를 보정한다.
    $$\hat{S}_u = \bar{S}_u (1 - d_{foci}(u))$$

이렇게 생성된 Saliency map과 Step 3의 Attention map을 결합하여, 주의 영역 내에서 가장 Saliency가 높은 지점을 **Foreground(FG) seed**로 설정한다. 이후, 그레이스케일 영상의 Gradient $\nabla I$를 분석하여 FG seed로부터 4방향으로 탐색하며 경계선(High gradient)을 지나친 지점의 픽셀들을 **Background(BG) seed**로 설정한다.

### Step 5: 병변 분할(Lesion Segmentation)
추출된 FG 및 BG seed를 입력값으로 하여 **Random Walk (RW)** 알고리즘을 적용한다. RW는 시드 지점들로부터 시작하여 픽셀 간의 유사도에 따라 확률적으로 영역을 확장함으로써 객체의 정밀한 공간적 범위를 결정한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Lung Tissue Research Consortium (LTRC)에서 제공한 폐암 환자의 흉부 CT 볼륨 4개.
- **피험자**: 경력 차이가 뚜렷한 3명의 방사선 전문의 (경력 20년, 10년, 3년).
- **평가 지표**: Dice Similarity Coefficient (DSC) 및 Hausdorff Distance (HD).

### 정량적 결과
- **평균 DSC**: $86\%$
- **평균 HD**: $1.45\text{mm}$
- 수동으로 시드를 지정한 Random Walk 방식과 비교했을 때, 통계적으로 유의미한 차이가 없음($p > 0.05$)이 확인되었다. 이는 시선 추적 기반의 자동 시드 추출이 수동 지정만큼 효과적임을 시사한다.

### 정성적 결과
- 전문의의 숙련도가 낮을수록(경력 3년) 탐색 범위가 넓고 주의 지점이 분산되는 경향을 보였으나, 결과적으로 병변 부위에서는 모든 전문의의 주의 지점이 겹치는 현상이 관찰되었다.
- `Gaze2Segment` 시스템은 사용자의 숙련도나 시선 패턴에 관계없이 주의 집중 영역을 정확히 포착하여 분할을 수행하였다.

## 🧠 Insights & Discussion

### 강점
본 연구는 인간의 인지 과정(시선 집중)을 컴퓨터 비전의 전처리 단계(Recognition)에 직접 통합함으로써, 의료 영상 분할의 고질적인 문제인 '객체 위치 찾기'를 효율적으로 해결하였다. 특히 전문의가 인지적으로 판단한 영역을 알고리즘이 자동으로 따라가게 함으로써 사용자 경험을 해치지 않고 분석을 수행할 수 있다는 점이 매우 강력하다.

### 한계 및 비판적 해석
1.  **데이터 부족**: 4개의 CT 볼륨이라는 매우 제한된 데이터셋을 사용하여 일반화 성능을 입증하기에는 부족함이 있다.
2.  **False Positives**: 전문의가 병변이 아님에도 불구하고 단순히 관심을 갖고 오래 응시한 영역(Non-lesion regions)이 병변으로 오인되어 분할되는 경우가 발생한다. 이는 시선 정보가 반드시 병변과 1:1 대응되는 것은 아니라는 생물학적 한계를 보여준다. 이를 해결하기 위해 CAD(Computer-Aided Detection) 시스템과의 결합이 필요해 보인다.
3.  **파라미터 의존성**: $\epsilon$이나 $\hat{t}$와 같은 주요 파라미터가 데이터 기반이 아닌 경험적(Empirical)으로 설정되었다. 이는 다른 장기나 다른 모달리티의 영상에 적용할 때 재조정(Tuning)이 필요함을 의미한다.
4.  **오프라인 처리**: 현재 시스템은 데이터 기록 후 사후에 분할을 수행한다. 실제 임상 적용을 위해서는 판독과 동시에 실시간으로 분할 결과가 제공되는 온라인 시스템으로의 발전이 필수적이다.

## 📌 TL;DR

본 논문은 방사선 전문의의 시선 추적 데이터를 이용해 병변의 위치를 인식하고, 이를 Saliency map 및 Gradient 정보와 결합해 자동으로 시드를 생성한 뒤 Random Walk 알고리즘으로 분할하는 `Gaze2Segment` 시스템을 제안하였다. 실험 결과 $86\%$의 DSC를 기록하며 시선 정보가 의료 영상의 자동 인식 전략으로 유효함을 입증하였다. 이 연구는 인간의 생물학적 주의 집중과 컴퓨터 비전을 통합한 선구적인 시도로, 향후 실시간 의료 영상 분석 도구 개발에 중요한 기반이 될 것으로 평가된다.