{{- define "ragkit.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ragkit.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "ragkit.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "ragkit.labels" -}}
app.kubernetes.io/name: {{ include "ragkit.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{- define "ragkit.selector" -}}
app.kubernetes.io/name: {{ include "ragkit.name" .root }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end -}}

{{- define "ragkit.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "ragkit.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/* Container settings shared by the API, workers and scheduler. */}}
{{- define "ragkit.container" -}}
image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
imagePullPolicy: {{ .Values.image.pullPolicy }}
securityContext:
  {{- toYaml .Values.securityContext | nindent 2 }}
envFrom:
  - secretRef:
      name: {{ .Values.existingSecret }}
env:
  - {name: HOME, value: /tmp}
  {{- with .Values.env }}
  {{- toYaml . | nindent 2 }}
  {{- end }}
volumeMounts:
  - {name: tmp, mountPath: /tmp}
  {{- if .Values.projectConfig }}
  - {name: project-config, mountPath: /project/ragkit.yaml, subPath: ragkit.yaml}
  {{- end }}
{{- end -}}

{{- define "ragkit.volumes" -}}
- name: tmp
  emptyDir: {}
{{- if .Values.projectConfig }}
- name: project-config
  configMap:
    name: {{ include "ragkit.fullname" . }}-config
{{- end }}
{{- end -}}

{{- define "ragkit.podSpec" -}}
serviceAccountName: {{ include "ragkit.serviceAccountName" . }}
securityContext:
  {{- toYaml .Values.podSecurityContext | nindent 2 }}
{{- with .Values.imagePullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 2 }}
{{- end }}
volumes:
  {{- include "ragkit.volumes" . | nindent 2 }}
{{- end -}}
