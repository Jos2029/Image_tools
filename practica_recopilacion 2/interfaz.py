import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import modelos_de_color as mc
import operaciones_logicas as ol
import etiquetado as et
import operaciones_morfologicas as om
import segmentacion as seg
import fourier
import edges
import ruido_y_filtros as rf
import morfologia_binaria as mb
import morfologia_lattice as ml


class Aplicacion:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision Lab - Procesamiento Avanzado de Imágenes")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e2e')
        
        self.imagen_original = None
        self.imagen_actual = None
        self.imagen_secundaria = None
        self.kernel_morfologico = "3x3 cuadrado"
        self.imagen_patron = None
        self.filtro_var = tk.StringVar(value="NINGUNO")


        self.operaciones_binarias = [
            "Erosión", "Dilatación", "Apertura", "Cierre",
            "Gradiente", "Frontera", "Hit-or-Miss",
            "Adelgazamiento", "Esqueleto", "Aislamiento"
        ]
        self.operaciones_lattice = [
            "Erosión", "Dilatación", "Apertura",
            "Cierre", "Gradiente"
        ]
        self.configurar_estilos()
        self.crear_interfaz()

      
    def configurar_estilos(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # ============================================
        # AQUÍ SE MODIFICAN TODOS LOS COLORES
        # ============================================
        bg_dark = '#1e1e2e'      # Fondo principal oscuro
        bg_card = '#2a2a3e'      # Fondo de las tarjetas
        accent = '#00d4ff'       # Color de acento (cyan)
        text_color = '#000000'   # Color de texto principal
        text_muted = '#b0b0b0'   # Color de texto secundario (más claro)
        btn_bg = '#4a4a6e'       # Fondo de botones (más claro para mejor contraste)
        btn_hover = '#5a5a7e'    # Color hover de botones
        
        style.configure('Dark.TFrame', background=bg_dark)
        style.configure('Card.TFrame', background=bg_card, relief='flat')
        style.configure('Header.TLabel', background=bg_dark, foreground=accent, 
                       font=('Segoe UI', 16, 'bold'))
        style.configure('Title.TLabel', background=bg_card, foreground=text_color, 
                       font=('Segoe UI', 11, 'bold'))
        style.configure('Accent.TButton', background=accent, foreground='#000000',
                       font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.configure('Dark.TLabel', background=bg_card, foreground=text_color,
                       font=('Segoe UI', 9))
        
        # Guardar colores para uso en widgets personalizados
        self.colores = {
            'bg_dark': bg_dark,
            'bg_card': bg_card,
            'accent': accent,
            'text_color': text_color,
            'text_muted': text_muted,
            'btn_bg': btn_bg,
            'btn_hover': btn_hover
        }
    
    def crear_interfaz(self):
        # Container principal
        container = tk.Frame(self.root, bg='#1e1e2e')
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # HEADER
        header = tk.Frame(container, bg='#1e1e2e', height=80)
        header.pack(fill=tk.X, pady=(0, 15))
        header.pack_propagate(False)
        
        ##itle_label = tk.Label(header, text="🎨 VISION LAB", 
          #                  font=('Segoe UI', 24, 'bold'),
           #                   fg='#00d4ff', bg='#1e1e2e')
        #title_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        subtitle = tk.Label(header, text="Herramienta profesional de análisis visual",
                           font=('Segoe UI', 10), fg='#888888', bg='#1e1e2e')
        #subtitle.pack(side=tk.LEFT, padx=(0, 20))
        
        # Botones de acción rápida en el header
        btn_frame = tk.Frame(header, bg='#1e1e2e')
        btn_frame.pack(side=tk.RIGHT, padx=20)
        
        self.crear_boton_header(btn_frame, "📁 Abrir", self.cargar_imagen).pack(side=tk.LEFT, padx=5)
        self.crear_boton_header(btn_frame, "💾 Guardar", self.guardar_imagen).pack(side=tk.LEFT, padx=5)
        self.crear_boton_header(btn_frame, "↺ Reset", self.restaurar_imagen).pack(side=tk.LEFT, padx=5)
        
        # AREA DE TRABAJO (3 columnas) - CON PESOS RESPONSIVOS
        workspace = tk.Frame(container, bg='#1e1e2e')
        workspace.pack(fill=tk.BOTH, expand=True)
        
        # Configurar grid para layout responsivo
        workspace.grid_rowconfigure(0, weight=1)
        #workspace.grid_columnconfigure(0, weight=0, minsize=280)  # Izquierda: tamaño fijo
        workspace.grid_columnconfigure(1, weight=1)                # Centro: expansible
        workspace.grid_columnconfigure(2, weight=0, minsize=600)  # Derecha: tamaño fijo
        
        
        # COLUMNA CENTRAL - Visualización
        center_panel = tk.Frame(workspace, bg='#1e1e2e')
        center_panel.grid(row=0, column=1, sticky='nsew', padx=10)
        
        self.crear_panel_visualizacion(center_panel)
        
        # COLUMNA DERECHA - Configuración
        right_panel = tk.Frame(workspace, bg='#1e1e2e', width=300)
        right_panel.grid(row=0, column=2, sticky='nsew', padx=(10, 0))
        
        self.crear_panel_configuracion(right_panel)
    
    def crear_boton_header(self, parent, text, command):
        btn = tk.Button(parent, text=text, command=command,
                       bg='#00d4ff', fg='#000000', 
                       font=('Segoe UI', 10, 'bold'),
                       relief='flat', cursor='hand2',
                       padx=20, pady=10)
        btn.bind('<Enter>', lambda e: e.widget.config(bg='#00b8e6'))
        btn.bind('<Leave>', lambda e: e.widget.config(bg='#00d4ff'))
        return btn
    
    def crear_card(self, parent, title):
        card = tk.Frame(parent, bg='#2a2a3e', relief='flat', bd=0)
        card.pack(fill=tk.X, pady=8)
        
        header = tk.Frame(card, bg='#2a2a3e', height=40)
        header.pack(fill=tk.X, padx=15, pady=(10, 5))
        
        tk.Label(header, text=title, font=('Segoe UI', 11, 'bold'),
                fg='#00d4ff', bg='#2a2a3e').pack(side=tk.LEFT)
        
        content = tk.Frame(card, bg='#2a2a3e')
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        return content
    
    def crear_boton_tool(self, parent, text, command):
        btn = tk.Button(parent, text=text, command=command,
                       bg=self.colores['btn_bg'], fg=self.colores['text_color'],
                       font=('Segoe UI', 9, 'bold'),  # Texto en negrita para mejor visibilidad
                       relief='flat', cursor='hand2',
                       padx=15, pady=10)  # Más padding vertical
        btn.pack(fill=tk.X, pady=3)
        btn.bind('<Enter>', lambda e: e.widget.config(bg=self.colores['btn_hover']))
        btn.bind('<Leave>', lambda e: e.widget.config(bg=self.colores['btn_bg']))
        return btn
    
    def crear_boton_seg(self, parent, text, command):
        """Botón específico para segmentación (más compacto)"""
        btn = tk.Button(parent, text=text, command=command,
                       bg='#3a3a5e', fg='#000000',
                       font=('Segoe UI', 8),
                       relief='flat', cursor='hand2',
                       padx=10, pady=6)
        btn.pack(fill=tk.X, pady=2)
        btn.bind('<Enter>', lambda e: e.widget.config(bg='#4a4a6e'))
        btn.bind('<Leave>', lambda e: e.widget.config(bg='#3a3a5e'))
        return btn
    
    def cargar_patron(self):
        ruta = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )
        if ruta:
            self.imagen_patron = cv2.imread(ruta)
            if self.imagen_patron is not None:
                messagebox.showinfo("✓ Patrón cargado", "Imagen patrón cargada correctamente")
            else:
                messagebox.showerror("❌ Error", "No se pudo cargar la imagen patrón")

    def crear_panel_visualizacion(self, parent):
        # Crear un Canvas con Scrollbar
        canvas = tk.Canvas(parent, bg='#1e1e2e', highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1e1e2e')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Vincular el desplazamiento con la rueda del mouse
        def _on_mousewheel(event):
            # Ajustar el desplazamiento según el sistema operativo
            if event.delta:  # Windows y algunos sistemas
                canvas.yview_scroll(-1 * (event.delta // 120), "units")
            else:  # macOS y otros sistemas Unix
                canvas.yview_scroll(-1 * event.num, "units")

        # En sistemas Unix (macOS y Linux), usa <Button-4> y <Button-5>
        def _on_mousewheel_unix(event):
            if event.num == 4:  # Scroll hacia arriba
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:  # Scroll hacia abajo
                canvas.yview_scroll(1, "units")

        # Vincular eventos de desplazamiento
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))  # Windows
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        canvas.bind_all("<Button-4>", _on_mousewheel_unix)  # Unix (scroll up)
        canvas.bind_all("<Button-5>", _on_mousewheel_unix)  # Unix (scroll down)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configurar grid para las imágenes
        scrollable_frame.grid_rowconfigure(0, weight=1)  # Fila para la imagen original
        scrollable_frame.grid_rowconfigure(1, weight=1)  # Fila para la imagen procesada
        scrollable_frame.grid_rowconfigure(2, weight=1)  # Fila para la imagen secundaria
        scrollable_frame.grid_columnconfigure(0, weight=1)  # Una sola columna

        # Imagen Original
        orig_card = tk.Frame(scrollable_frame, bg='#2a2a3e')
        orig_card.grid(row=0, column=0, sticky='nsew', padx=10, pady=5)

        tk.Label(orig_card, text="ORIGINAL", 
                font=('Segoe UI', 10, 'bold'),
                fg='#888888', bg='#2a2a3e').pack(pady=10)

        self.label_original = tk.Label(orig_card, bg='#1a1a2a', anchor='center')
        self.label_original.pack(fill=tk.NONE, expand=True, padx=15, pady=(0, 15))

        # Imagen Procesada
        proc_card = tk.Frame(scrollable_frame, bg='#2a2a3e')
        proc_card.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)

        tk.Label(proc_card, text="PROCESADA", 
                font=('Segoe UI', 10, 'bold'),
                fg='#00d4ff', bg='#2a2a3e').pack(pady=10)

        self.label_procesada = tk.Label(proc_card, bg='#1a1a2a', anchor='center')
        self.label_procesada.pack(fill=tk.NONE, expand=True, padx=15, pady=(0, 15))

        # Imagen Secundaria
        sec_card = tk.Frame(scrollable_frame, bg='#2a2a3e')
        sec_card.grid(row=2, column=0, sticky='nsew', padx=10, pady=5)

        tk.Label(sec_card, text="SECUNDARIA", 
                font=('Segoe UI', 10, 'bold'),
                fg='#ff8800', bg='#2a2a3e').pack(pady=10)

        self.label_secundaria = tk.Label(sec_card, bg='#1a1a2a', anchor='center')
        self.label_secundaria.pack(fill=tk.NONE, expand=True, padx=15, pady=(0, 15))


    def crear_panel_configuracion(self, parent):
        # Canvas con scroll vertical Y horizontal
        canvas = tk.Canvas(parent, bg='#1e1e2e', highlightthickness=0)
        scrollbar_v = tk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollbar_h = tk.Scrollbar(parent, orient='horizontal', command=canvas.xview)
        scrollable_frame = tk.Frame(canvas, bg='#1e1e2e')
        
        scrollable_frame.bind('<Configure>', 
                             lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)

        # Vincular el desplazamiento con la rueda del mouse
        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * (event.delta // 120), "units")
        def _on_mousewheel_unix(event):
            if event.num == 4:  # Scroll hacia arriba
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:  # Scroll hacia abajo
                canvas.yview_scroll(1, "units")
        
        # Vincular eventos de desplazamiento
        canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows
        canvas.bind_all("<Button-4>", _on_mousewheel_unix)  # Unix (scroll up)
        canvas.bind_all("<Button-5>", _on_mousewheel_unix)  # Unix (scroll down)
            
        tk.Label(scrollable_frame, text="⚙️ Configuración", 
                font=('Segoe UI', 14, 'bold'),
                fg='#00d4ff', bg='#1e1e2e').pack(pady=(0, 15))
        
        # CARD: Operaciones Lógicas
        logic_content = self.crear_card(scrollable_frame, "⚡ Operaciones Lógicas")
        
        self.crear_boton_tool(logic_content, "📂 Cargar 2ª Imagen", self.cargar_secundaria)
        
        ops_frame = tk.Frame(logic_content, bg='#2a2a3e')
        ops_frame.pack(fill=tk.X, pady=5)
        
        for op in ["AND", "OR", "XOR", "NOT"]:
            tk.Button(ops_frame, text=op, 
                     command=lambda o=op: self.operacion_logica(o),
                     bg='#4a4a5e', fg='#000000', font=('Segoe UI', 8, 'bold'),
                     relief='flat', cursor='hand2', width=6, pady=5).pack(side=tk.LEFT, padx=2)

        # CARD: Morfología
        morph_content = self.crear_card(scrollable_frame, "🧬 Morfología")
        tk.Label(morph_content, text="Tipo de Morfología",
        fg='#888888', bg='#2a2a3e', font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 3))

        self.morph_op_var = tk.StringVar()
        self.morph_combo = ttk.Combobox(
            morph_content,
            textvariable=self.morph_op_var,
            state="readonly",
            font=('Segoe UI', 9)
        )
        self.morph_combo.pack(fill=tk.X, pady=(0, 15))


        self.tipo_morf_var = tk.StringVar(value="Automática")

        for tipo in ["Binaria", "Lattice", "Automática"]:
            tk.Radiobutton(
                morph_content,
                text=tipo,
                variable=self.tipo_morf_var,
                value=tipo,
                bg='#2a2a3e',
                fg='#ffffff',
                selectcolor='#3a3a4e',
                font=('Segoe UI', 9)
            ).pack(anchor=tk.W)


        
        # Tipo de Kernel
        tk.Label(morph_content, text="Tipo de Kernel", 
                fg='#888888', bg='#2a2a3e', font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 3))
        
        self.kernel_var = tk.StringVar(value="3x3 cuadrado")
        kernel_combo = ttk.Combobox(morph_content, textvariable=self.kernel_var,
                                   values=["3x3 cuadrado", "5x5 cuadrado", "7x7 cuadrado",
                                          "3x3 eliptico", "5x5 eliptico",
                                          "3x3 cruz", "5x5 cruz"],
                                   state="readonly", font=('Segoe UI', 9))
        kernel_combo.pack(fill=tk.X, pady=(0, 10))
        kernel_combo.bind('<<ComboboxSelected>>', self.actualizar_kernel)
        
        # Operación Morfológica
        
        self.tipo_morf_var.trace_add("write", self.actualizar_operaciones_morf)
        self.actualizar_operaciones_morf()

        
        # Botón Aplicar grande
        apply_btn = tk.Button(morph_content, text="▶ APLICAR MORFOLOGÍA",
                             command=self.aplicar_morph,
                             bg='#00d4ff', fg='#000000',
                             font=('Segoe UI', 10, 'bold'),
                             relief='flat', cursor='hand2', pady=12)
        apply_btn.pack(fill=tk.X)
        apply_btn.bind('<Enter>', lambda e: e.widget.config(bg='#00b8e6'))
        apply_btn.bind('<Leave>', lambda e: e.widget.config(bg='#00d4ff'))
        

        # CARD: Transformaciones de Color
        color_content = self.crear_card(scrollable_frame, "🎨 Color & Filtros")
        
        self.crear_boton_tool(color_content, "▸ Escala de Grises", self.grises)
        self.crear_boton_tool(color_content, "▸ Binarizar (128)", self.binarizar)
        self.crear_boton_tool(color_content, "▸ Binarizar Custom", self.binarizar_con_umbral)
        self.crear_boton_tool(color_content,"Calcular Histograma", self.mostrar_histograma)
        
        tk.Label(color_content, text="Espacio de color:", 
                fg='#ffffff', bg='#2a2a3e', 
                font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W, pady=(10, 3))
        
        self.modelo_var = tk.StringVar(value="RGB")
        for modelo in ["RGB", "RGB_CANALES", "CMYK", "HSV", "PSEUDOCOLOR PASTEL", "PSEUDOCOLOR TIERRA", "PSEUDOCOLOR FRIOS"]:
            rb = tk.Radiobutton(color_content, text=modelo, variable=self.modelo_var,
                               value=modelo, command=self.cambiar_modelo_color,
                               bg='#2a2a3e', fg='#ffffff', 
                               selectcolor='#3a3a4e',
                               activebackground='#2a2a3e', activeforeground='#00d4ff',
                               font=('Segoe UI', 9, 'bold'))  # Bold para mejor visibilidad
            rb.pack(anchor=tk.W, padx=5, pady=2)
        
        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar_v.grid(row=0, column=1, sticky='ns')
        scrollbar_h.grid(row=1, column=0, sticky='ew')

        # CARD: Segmentación
        seg_content = self.crear_card(scrollable_frame, "✂️ Segmentación")
        
        tk.Label(seg_content, text="Métodos de Umbralización", 
                fg='#ffffff', bg='#2a2a3e', font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W, pady=(5, 5))
        
        self.crear_boton_seg(seg_content, "Otsu", self.seg_otsu)
        self.crear_boton_seg(seg_content, "Media", self.seg_media)
        self.crear_boton_seg(seg_content, "Kapur", self.seg_kapur)
        self.crear_boton_seg(seg_content, "Comparar Histogramas", self.comparar_histogramas)

        # =========================
        # BOTONES DE FIGURAS
        # =========================
        self.crear_boton_seg(seg_content, "Detectar Figuras", self.detectar_figuras)
        self.crear_boton_seg(seg_content, "Comparar Segmentaciones", self.comparar_segmentaciones)

        
        tk.Label(seg_content, text="Ajuste de Brillo", 
                fg='#ffffff', bg='#2a2a3e', font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W, pady=(15, 5))
        
        self.crear_boton_seg(seg_content, "Ecualización Uniforme", self.seg_eq_uniforme)
        self.crear_boton_seg(seg_content, "Ecualización Exponencial", self.seg_eq_exponencial)
        self.crear_boton_seg(seg_content, "Ecualización Rayleigh", self.seg_eq_rayleigh)
        self.crear_boton_seg(seg_content, "Ecualización Hipercúbica", self.seg_eq_hipercubica)
        self.crear_boton_seg(seg_content, "Ecualización Logarítmica", self.seg_eq_logaritmica)
        self.crear_boton_seg(seg_content, "Corrección Gamma", self.seg_gamma)
        self.crear_boton_seg(seg_content, "Watershed", self.seg_watershed)

        

        # CARD # CARD: FILTROS
        filtros_content = self.crear_card(scrollable_frame, "🧪 Filtros")

        self.crear_boton_seg(filtros_content, "Ruido Sal y Pimienta", self.filtro_ruido_sp)
        self.crear_boton_seg(filtros_content, "Ruido Gaussiano", self.filtro_ruido_gauss)

        self.crear_boton_seg(filtros_content, "Promediador", self.filtro_promediador)
        self.crear_boton_seg(filtros_content, "Promediador Pesado", self.filtro_prom_pesado)
        self.crear_boton_seg(filtros_content, "Gaussiano", self.filtro_gaussiano)
        self.crear_boton_seg(filtros_content, "Laplaciano", self.filtro_laplaciano)

        self.crear_boton_seg(filtros_content, "Mediana", self.filtro_mediana)
        self.crear_boton_seg(filtros_content, "Moda", self.filtro_moda)
        self.crear_boton_seg(filtros_content, "Máximo", self.filtro_maximo)
        self.crear_boton_seg(filtros_content, "Mínimo", self.filtro_minimo)
        self.crear_boton_seg(filtros_content, "Bilateral", self.filtro_bilateral)
        
        
        
        # CARD: BORDES
        edges_content = self.crear_card(scrollable_frame, "🧠 Detección de Bordes")

        self.crear_boton_seg(edges_content, "Sobel", self.edge_sobel)
        self.crear_boton_seg(edges_content, "Prewitt", self.edge_prewitt)
        self.crear_boton_seg(edges_content, "Roberts", self.edge_roberts)
        self.crear_boton_seg(edges_content, "Canny", self.edge_canny)

        # CARD: FOURIER
        fourier_content = self.crear_card(scrollable_frame, "📊 Transformada de Fourier")

        self.crear_boton_seg(fourier_content, "FFT (Espectro)", self.fourier_fft)
        self.crear_boton_seg(fourier_content, "Pasa Bajas", self.fourier_low_pass)
        self.crear_boton_seg(fourier_content, "Pasa Altas", self.fourier_high_pass)

        # Info adicional
        info_frame = tk.Frame(scrollable_frame, bg='#2a2a3e')
        info_frame.pack(fill=tk.X, pady=(20, 0))
        
        tk.Label(info_frame, text="ℹ️ Información", 
                font=('Segoe UI', 10, 'bold'),
                fg='#000000', bg='#2a2a3e').pack(pady=10, padx=15)
        
        self.info_text = tk.Text(info_frame, height=6, bg='#1a1a2a', fg='#888888',
                                font=('Consolas', 8), relief='flat', padx=10, pady=10,
                                wrap=tk.WORD)
        self.info_text.pack(fill=tk.X, padx=15, pady=(0, 15))
        self.info_text.insert('1.0', 'Esperando imagen...\n\n• Carga una imagen para comenzar\n• Aplica filtros y transformaciones\n• Compara resultados en tiempo real')
        self.info_text.config(state='disabled')
        
        # Empaquetar scrollbars
        canvas.grid(row=0, column=0, sticky='nsew')
        scrollbar_v.grid(row=0, column=1, sticky='ns')
        scrollbar_h.grid(row=1, column=0, sticky='ew')
        
        # Configurar grid del parent
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        # MOSTRAR canvas y scrollbar
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar_v.grid(row=0, column=1, sticky="ns")
        scrollbar_h.grid(row=1, column=0, sticky='ew')


    
    # Métodos de procesamiento (mantienen la misma lógica)
    def actualizar_kernel(self, event=None):
        self.kernel_morfologico = self.kernel_var.get()
    
    def obtener_configuracion_kernel(self):
        config = self.kernel_morfologico.split()
        tamaño = int(config[0][0])
        tipo = config[1]
        
        if tipo == "cuadrado":
            tipo_kernel = "cuadrado"
        elif tipo == "eliptico":
            tipo_kernel = "eliptico"
        elif tipo == "cruz":
            tipo_kernel = "cruz"
        else:
            tipo_kernel = "cuadrado"
        
        return tipo_kernel, tamaño
   
    def es_binaria(self, imagen):
        if imagen is None:
            return False
        if len(imagen.shape) != 2:
            return False
        valores = np.unique(imagen)
        return np.array_equal(valores, [0, 255]) \
            or np.array_equal(valores, [0]) \
            or np.array_equal(valores, [255])

    def aplicar_morph(self):
        if not self.morph_op_var.get():
            messagebox.showwarning("⚠️ Advertencia", "Seleccione una operación morfológica")
            return

        if self.imagen_actual is None:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
            return

        tipo_kernel, tamaño = self.obtener_configuracion_kernel()
        operacion = self.morph_op_var.get()
        tipo_sel = self.tipo_morf_var.get()

        # Convertir a gris para procesar
        img = self.imagen_actual
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Selección de morfología
        if tipo_sel == "Binaria":
            _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            modulo = mb
            tipo_morf = "Binaria"
        elif tipo_sel == "Lattice":
            modulo = ml
            tipo_morf = "Lattice"
        else:  # Automática
            if self.es_binaria(img):
                modulo = mb
                tipo_morf = "Binaria (auto)"
            else:
                modulo = ml
                tipo_morf = "Lattice (auto)"

        # Operación
        if operacion == "Erosión":
            resultado = modulo.erosion(img, tipo_kernel, tamaño)
        elif operacion == "Dilatación":
            resultado = modulo.dilatacion(img, tipo_kernel, tamaño)
        elif operacion == "Apertura":
            resultado = modulo.apertura(img, tipo_kernel, tamaño)
        elif operacion == "Cierre":
            resultado = modulo.cierre(img, tipo_kernel, tamaño)
        elif operacion == "Gradiente":
            resultado = modulo.gradiente(img, tipo_kernel, tamaño)
        elif operacion == "Frontera":
            resultado = modulo.frontera(img, tipo_kernel, tamaño)

        elif operacion == "Hit-or-Miss":
            resultado = modulo.hit_or_miss(img)

        elif operacion == "Adelgazamiento":
            resultado = modulo.adelgazamiento(img)

        elif operacion == "Esqueleto":
            resultado = modulo.esqueleto(img)

        elif operacion == "Aislamiento":
            resultado = modulo.aislamiento(img)


        self.imagen_actual = resultado
        self.actualizar_info(f"Morfología {tipo_morf}: {operacion} ({self.kernel_morfologico})")
        self.mostrar_imagenes()
   
   
    def actualizar_operaciones_morf(self, *args):
            tipo = self.tipo_morf_var.get()

            if tipo == "Binaria":
                self.morph_combo["values"] = self.operaciones_binarias
                self.morph_op_var.set(self.operaciones_binarias[0])

            elif tipo == "Lattice":
                self.morph_combo["values"] = self.operaciones_lattice
                self.morph_op_var.set(self.operaciones_lattice[0])

            else:  # Automática
                self.morph_combo["values"] = self.operaciones_binarias
                self.morph_op_var.set(self.operaciones_binarias[0])

    
    def actualizar_info(self, mensaje):
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', f'✓ {mensaje}\n\n{self.obtener_info_imagen()}')
        self.info_text.config(state='disabled')
    
    def obtener_info_imagen(self):
        if self.imagen_actual is not None:
            h, w = self.imagen_actual.shape[:2]
            canales = self.imagen_actual.shape[2] if len(self.imagen_actual.shape) == 3 else 1
            return f"Dimensiones: {w}x{h}\nCanales: {canales}\nTipo: {'Color' if canales > 1 else 'Gris'}"
        return "No hay imagen cargada"
    
    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if ruta:
            self.imagen_original = cv2.imread(ruta)
            if self.imagen_original is not None:
                self.imagen_actual = self.imagen_original.copy()
                self.actualizar_info("Imagen cargada correctamente")
                self.mostrar_imagenes()
            else:
                messagebox.showerror("❌ Error", "No se pudo cargar la imagen")
    
    def cargar_secundaria(self):
        ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if ruta:
            self.imagen_secundaria = cv2.imread(ruta)
            if self.imagen_secundaria is None:
                messagebox.showerror("❌ Error", "No se pudo cargar la imagen secundaria")
            else:
                messagebox.showinfo("✓ Éxito", "Imagen secundaria cargada correctamente")
                self.mostrar_imagenes()
    
    def mostrar_imagenes(self):
        # Calcular tamaño dinámico basado en el tamaño de la ventana
        self.root.update_idletasks()
        
        # Tamaño máximo para las imágenes (se ajusta al espacio disponible)
        max_width = 500
        max_height = 500
        
        # Mostrar imagen original
        if self.imagen_original is not None:  # Verificar si la imagen original existe
            img_orig = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB)
            img_orig = Image.fromarray(img_orig)
            img_orig.thumbnail((max_width, max_height), Image.LANCZOS)
            img_orig = ImageTk.PhotoImage(img_orig)
            self.label_original.configure(image=img_orig)
            self.label_original.image = img_orig
        
        # Mostrar imagen procesada
        if self.imagen_actual is not None:
            if len(self.imagen_actual.shape) == 2:
                img_actual = cv2.cvtColor(self.imagen_actual, cv2.COLOR_GRAY2RGB)
            else:
                img_actual = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2RGB)
        else:
            img_actual = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB)
            
        img_actual = Image.fromarray(img_actual)
        img_actual.thumbnail((max_width, max_height), Image.LANCZOS)
        img_actual = ImageTk.PhotoImage(img_actual)
        self.label_procesada.configure(image=img_actual)
        self.label_procesada.image = img_actual

        # Mostrar imagen secundaria
        if self.imagen_secundaria is not None:
            img_sec = cv2.cvtColor(self.imagen_secundaria, cv2.COLOR_BGR2RGB)
            img_sec = Image.fromarray(img_sec)
            img_sec.thumbnail((max_width, max_height), Image.LANCZOS)
            img_sec = ImageTk.PhotoImage(img_sec)
            self.label_secundaria.configure(image=img_sec)
            self.label_secundaria.image = img_sec
    
    def restaurar_imagen(self):
        if self.imagen_original is not None:
            self.imagen_actual = self.imagen_original.copy()
            self.actualizar_info("Imagen restaurada al original")
            self.mostrar_imagenes()
    
    def guardar_imagen(self):
        if self.imagen_actual is not None:
            ruta = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
            if ruta:
                cv2.imwrite(ruta, self.imagen_actual)
                messagebox.showinfo("✓ Éxito", "Imagen guardada correctamente")
        else:
            messagebox.showwarning("⚠️ Advertencia", "No hay imagen procesada para guardar")
    
    def binarizar(self):
        if self.imagen_actual is not None:
            self.imagen_actual = mc.binarizar_imagen(self.imagen_actual)
            self.actualizar_info("Binarización aplicada (umbral 128)")
            self.mostrar_imagenes()
    
    def binarizar_con_umbral(self):
        if self.imagen_actual is not None:
            umbral = simpledialog.askinteger("Binarización", 
                                           "Ingrese el valor de umbral (0-255):", 
                                           minvalue=0, maxvalue=255, initialvalue=128)
            if umbral is not None:
                self.imagen_actual = mc.binarizar_imagen_umbral(self.imagen_actual, umbral)
                self.actualizar_info(f"Binarización aplicada (umbral {umbral})")
                self.mostrar_imagenes()
    
    def grises(self):
        if self.imagen_actual is not None:
            self.imagen_actual = mc.escala_grises(self.imagen_actual)
            self.actualizar_info("Convertida a escala de grises")
            self.mostrar_imagenes()
    


    def cambiar_modelo_color(self):
        if self.imagen_actual is not None:
            modelo = self.modelo_var.get()

            # Caso especial: RGB por canales (solo visualización)
            if modelo == "RGB_CANALES":
                mc.mostrar_canales_rgb(self.imagen_actual)
                self.actualizar_info("Visualización de canales RGB (R, G, B)")
                return  # NO modifica imagen_actual

            # Otros modelos sí modifican la imagen
            self.imagen_actual = mc.aplicar_modelo_color(self.imagen_actual, modelo)
            self.actualizar_info(f"Modelo de color cambiado a {modelo}")
            self.mostrar_imagenes()

    # ========================= Aplicar Ruido y Filtros =========================

    def aplicar_ruido_filtro(self):
        if self.imagen_actual is None:
            return

        img = self.imagen_actual.copy()
        opcion = self.filtro_var.get()

        if opcion == "RUIDO_SAL_PIMIENTA":
            img = rf.ruido_sal_pimienta(img, 0.02)

        elif opcion == "RUIDO_GAUSSIANO":
            img = rf.ruido_gaussiano(img, sigma=25)

        elif opcion == "PROMEDIADOR":
            img = rf.filtro_promediador(img, 5)

        elif opcion == "GAUSSIANO":
            img = rf.filtro_gaussiano(img, 5)

        elif opcion == "MEDIANA":
            img = rf.filtro_mediana(img, 5)

        elif opcion == "SOBEL":
            img = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
            img = np.uint8(np.absolute(img))

        elif opcion == "LAPLACIANO":
            img = rf.filtro_laplaciano(img)

        self.mostrar_imagenes(img)

    
    def operacion_logica(self, op):
        if self.imagen_actual is not None:
            if op == "NOT":
                self.imagen_actual = ol.operacion_not(self.imagen_actual)
                self.actualizar_info("Operación NOT aplicada")
            else:
                if self.imagen_secundaria is None:
                    messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen secundaria")
                    return
                
                if self.imagen_actual.shape != self.imagen_secundaria.shape:
                    self.imagen_secundaria = cv2.resize(self.imagen_secundaria, 
                                                       (self.imagen_actual.shape[1], 
                                                        self.imagen_actual.shape[0]))
                
                if op == "AND":
                    self.imagen_actual = ol.operacion_and(self.imagen_actual, self.imagen_secundaria)
                elif op == "OR":
                    self.imagen_actual = ol.operacion_or(self.imagen_actual, self.imagen_secundaria)
                elif op == "XOR":
                    self.imagen_actual = ol.operacion_xor(self.imagen_actual, self.imagen_secundaria)
                
                self.actualizar_info(f"Operación {op} aplicada")
            
            self.mostrar_imagenes()
    # =========================
    # MÉTODOS DE ANÁLISIS DE REGIONES
    # =========================
    def extraccion_umbral(self):
        if self.imagen_actual is not None:
            umbral = simpledialog.askinteger("Umbral", "Ingrese el valor de umbral (0-255)", 
                                           minvalue=0, maxvalue=255, initialvalue=128)
            if umbral is not None:
                binaria = et.extraer_regiones_umbral(self.imagen_actual, umbral, 255)
                self.imagen_actual = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)
                self.actualizar_info(f"Extracción por umbral ({umbral})")
                self.mostrar_imagenes()
    def mostrar_histograma(self):
        if self.imagen_actual is None:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
            return

        # Escala de grises
        if len(self.imagen_actual.shape) == 3:
            gris = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
        else:
            gris = self.imagen_actual

        # Histograma
        hist = cv2.calcHist([gris], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()  # Normalizado

        niveles = np.arange(256)

        # === PROPIEDADES ===
        media = np.sum(niveles * hist_norm)
        varianza = np.sum(((niveles - media) ** 2) * hist_norm)
        desviacion = np.sqrt(varianza)
        entropia = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
        energia = np.sum(hist_norm ** 2)

        # Mostrar histograma
        plt.figure("Histograma de la imagen")
        plt.plot(hist, color='black')
        plt.title("Histograma de niveles de gris")
        plt.xlabel("Nivel de intensidad")
        plt.ylabel("Frecuencia")
        plt.grid(True)

        texto = (
            f"Media: {media:.2f}\n"
            f"Varianza: {varianza:.2f}\n"
            f"Desv. Std: {desviacion:.2f}\n"
            f"Entropía: {entropia:.2f}\n"
            f"Energía: {energia:.4f}"
        )

        plt.text(
            170, max(hist)*0.7,
            texto,
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8)
        )

        plt.show()


        # Mostrar propiedades en el panel de info
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert(
            '1.0',
            f"✓ Propiedades del histograma\n\n"
            f"Media: {media:.2f}\n"
            f"Varianza: {varianza:.2f}\n"
            f"Desviación estándar: {desviacion:.2f}\n"
            f"Entropía: {entropia:.2f}\n"
            f"Energía: {energia:.4f}"
        )
        self.info_text.config(state='disabled')


    def etiquetar_regiones(self):
        if self.imagen_actual is not None:
            binaria = et.extraer_regiones_umbral(self.imagen_actual, 128, 255)
            self.imagen_actual = et.etiquetar_regiones(binaria)
            self.actualizar_info("Regiones etiquetadas")
            self.mostrar_imagenes()
   
    def etiquetar_patron(self):
        if self.imagen_actual is None or self.imagen_patron is None:
            messagebox.showwarning(
                "Advertencia", "Debe cargar imagen y patrón"
            )
            return

        binaria = et.extraer_regiones_umbral(self.imagen_actual, 128, 255)
        patron_bin = et.extraer_regiones_umbral(self.imagen_patron, 128, 255)

        self.imagen_actual = et.etiquetar_patron(binaria, patron_bin)
        self.actualizar_info("Reconocimiento de patrón por forma")
        self.mostrar_imagenes()


    
    # =========================
    # MÉTODOS DE SEGMENTACIÓN
    # =========================
    def seg_otsu(self):
        if self.imagen_actual is not None:
            self.imagen_actual = seg.umbral_otsu(self.imagen_actual)
            self.actualizar_info("Segmentación: Método Otsu")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def seg_media(self):
        if self.imagen_actual is not None:
            self.imagen_actual = seg.umbral_media(self.imagen_actual)
            self.actualizar_info("Segmentación: Umbral Media")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def seg_kapur(self):
        if self.imagen_actual is not None:
            self.imagen_actual = seg.umbral_kapur(self.imagen_actual)
            self.actualizar_info("Segmentación: Método Kapur")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def seg_eq_uniforme(self):
        if self.imagen_actual is not None:
            self.imagen_actual = seg.ecualizacion_uniforme(self.imagen_actual)
            self.actualizar_info("Ecualización Uniforme aplicada")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def seg_eq_exponencial(self):
        if self.imagen_actual is not None:
            self.imagen_actual = seg.ecualizacion_exponencial(self.imagen_actual)
            self.actualizar_info("Ecualización Exponencial aplicada")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def seg_eq_rayleigh(self):
        if self.imagen_actual is not None:
            self.imagen_actual = seg.ecualizacion_rayleigh(self.imagen_actual)
            self.actualizar_info("Ecualización Rayleigh aplicada")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def seg_eq_hipercubica(self):
        if self.imagen_actual is not None:
            self.imagen_actual = seg.ecualizacion_hipercubica(self.imagen_actual)
            self.actualizar_info("Ecualización Hipercúbica aplicada")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def seg_eq_logaritmica(self):
        if self.imagen_actual is not None:
            self.imagen_actual = seg.ecualizacion_logaritmica(self.imagen_actual)
            self.actualizar_info("Ecualización Logarítmica aplicada")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def seg_gamma(self):
        if self.imagen_actual is not None:
            gamma = simpledialog.askfloat("Corrección Gamma", 
                                         "Ingrese el valor de gamma (0.5 - 3.0):",
                                         minvalue=0.5, maxvalue=3.0, initialvalue=1.5)
            if gamma is not None:
                self.imagen_actual = seg.correccion_gamma(self.imagen_actual, gamma)
                self.actualizar_info(f"Corrección Gamma aplicada (γ={gamma})")
                self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    def seg_watershed(self):
        if self.imagen_actual is not None:
            binaria = seg.watershed_segmentacion(self.imagen_actual)
            self.imagen_actual = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)
            self.actualizar_info("Segmentación: Watershed + Transformada de Distancia")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")

    def mostrar_histograma(self):
        if self.imagen_actual is not None:
            seg.histograma(self.imagen_actual, titulo="Histograma de la imagen actual")
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")

    def comparar_histogramas(self):
        if self.imagen_actual is not None and self.imagen_original is not None:
            seg.comparacion_histogramas(self.imagen_original, self.imagen_actual)
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen y tener una original para comparar")

    def detectar_figuras(self):
        """
        Detecta figuras geométricas en la imagen binarizada actual,
        dibuja los contornos y etiquetas, y actualiza la imagen y la info.
        """
        if self.imagen_actual is None:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
            return

        # Copiar la imagen actual
        img = self.imagen_actual.copy()

        # Asegurarse que esté en escala de grises
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Encontrar contornos
        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Preparar imagen de resultado en color para dibujar
        resultado = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Diccionario de conteo de figuras
        conteo = {
            "Triangulo": 0,
            "Cuadrado": 0,
            "Rectangulo": 0,
            "Pentagono": 0,
            "Hexagono": 0,
            "Heptagono": 0,
            "Octagono": 0,
            "Circulo": 0,
            "Elipse": 0,
            "Estrella": 0
        }

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < 300:  # Ignorar contornos muy pequeños
                continue

            peri = cv2.arcLength(cnt, True)
            epsilon = 0.03 * peri
            aprox = cv2.approxPolyDP(cnt, epsilon, True)
            v = len(aprox)
            figura = None

            # Identificación de figuras
            if v == 3:
                figura = "Triangulo"
            elif v == 4:
                x, y, w, h = cv2.boundingRect(aprox)
                ar = w / h
                figura = "Cuadrado" if 0.9 <= ar <= 1.1 else "Rectangulo"
            elif v == 5:
                figura = "Pentagono"
            elif v == 6:
                figura = "Hexagono"
            elif v == 7:
                figura = "Heptagono"
            elif v == 8:
                figura = "Octagono"
            elif v > 8:
                circularidad = 4 * np.pi * area / (peri * peri)
                if circularidad > 0.8:
                    figura = "Circulo"
                elif circularidad > 0.65:
                    figura = "Elipse"
                else:
                    figura = "Estrella"

            # Dibujar contorno y etiqueta si se identificó
            if figura and figura in conteo:
                conteo[figura] += 1
                cv2.drawContours(resultado, [aprox], -1, (255,255,255), 2)  # Contorno blanco
                cv2.putText(resultado, figura, tuple(aprox[0][0]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)   # Texto verde

        # Actualizar imagen actual y mostrar información
        self.imagen_actual = resultado
        self.actualizar_info(f"Figuras detectadas: {conteo}")
        self.mostrar_imagenes()



    def comparar_segmentaciones(self):
        if self.imagen_actual is not None:
            segs = seg.comparar_segmentaciones(self.imagen_actual)
            self.actualizar_info("Comparación de segmentaciones (Otsu, Kapur, Media)")
            # opcional: podrías guardar o mostrar alguna de las segmentaciones
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")




    
    # =========================
    # MÉTODOS DE DETECCIÓN DE BORDES
    # =========================
    def edge_sobel(self):
        if self.imagen_actual is not None:
            # Convertir a escala de grises si es necesario
            if len(self.imagen_actual.shape) == 3:
                gris = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                gris = self.imagen_actual
            
            resultado = edges.sobel(gris)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info("Detección de bordes: Sobel")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def edge_prewitt(self):
        if self.imagen_actual is not None:
            if len(self.imagen_actual.shape) == 3:
                gris = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                gris = self.imagen_actual
            
            resultado = edges.prewitt(gris)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info("Detección de bordes: Prewitt")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def edge_roberts(self):
        if self.imagen_actual is not None:
            if len(self.imagen_actual.shape) == 3:
                gris = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                gris = self.imagen_actual
            
            resultado = edges.roberts(gris)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info("Detección de bordes: Roberts")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def edge_canny(self):
        if self.imagen_actual is not None:
            # Diálogo para umbrales
            t1 = simpledialog.askinteger("Canny - Umbral 1", 
                                        "Ingrese el umbral inferior (0-255):",
                                        minvalue=0, maxvalue=255, initialvalue=50)
            if t1 is None:
                return
            
            t2 = simpledialog.askinteger("Canny - Umbral 2", 
                                        "Ingrese el umbral superior (0-255):",
                                        minvalue=0, maxvalue=255, initialvalue=150)
            if t2 is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                gris = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                gris = self.imagen_actual
            
            resultado = edges.canny(gris, t1, t2)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Detección de bordes: Canny (t1={t1}, t2={t2})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    # =========================
    # MÉTODOS DE TRANSFORMADA DE FOURIER
    # =========================
    def fourier_fft(self):
        if self.imagen_actual is not None:
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            magnitude = fourier.compute_fft(img_input)
            # Normalizar para visualización
            magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude_norm = magnitude_norm.astype(np.uint8)
            
            self.imagen_actual = cv2.cvtColor(magnitude_norm, cv2.COLOR_GRAY2BGR)
            self.actualizar_info("FFT: Espectro de frecuencias")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def fourier_low_pass(self):
        if self.imagen_actual is not None:
            radius = simpledialog.askinteger("Filtro Pasa Bajas", 
                                           "Ingrese el radio del filtro (10-200):",
                                           minvalue=10, maxvalue=200, initialvalue=30)
            if radius is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = fourier.low_pass_filter(img_input, radius)
            resultado_norm = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
            resultado_norm = resultado_norm.astype(np.uint8)
            
            self.imagen_actual = cv2.cvtColor(resultado_norm, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Filtro Pasa Bajas aplicado (radio={radius})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def fourier_high_pass(self):
        if self.imagen_actual is not None:
            radius = simpledialog.askinteger("Filtro Pasa Altas", 
                                           "Ingrese el radio del filtro (10-200):",
                                           minvalue=10, maxvalue=200, initialvalue=30)
            if radius is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = fourier.high_pass_filter(img_input, radius)
            resultado_norm = cv2.normalize(resultado, None, 0, 255, cv2.NORM_MINMAX)
            resultado_norm = resultado_norm.astype(np.uint8)
            
            self.imagen_actual = cv2.cvtColor(resultado_norm, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Filtro Pasa Altas aplicado (radio={radius})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    # =========================
    # MÉTODOS DE FILTROS
    # =========================
    
    # --- AGREGAR RUIDO ---
    def filtro_ruido_sp(self):
        """Agregar ruido sal y pimienta"""
        if self.imagen_actual is not None:
            cantidad = simpledialog.askfloat("Ruido Sal y Pimienta", 
                                           "Ingrese la cantidad de ruido (0.01-0.5):",
                                           minvalue=0.01, maxvalue=0.5, initialvalue=0.02)
            if cantidad is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = rf.ruido_sal_pimienta(img_input, cantidad)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Ruido Sal y Pimienta agregado ({cantidad*100:.1f}%)")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def filtro_ruido_gauss(self):
        """Agregar ruido gaussiano"""
        if self.imagen_actual is not None:
            sigma = simpledialog.askinteger("Ruido Gaussiano", 
                                          "Ingrese sigma (desviación estándar) (10-100):",
                                          minvalue=10, maxvalue=100, initialvalue=25)
            if sigma is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = rf.ruido_gaussiano(img_input, sigma=sigma)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Ruido Gaussiano agregado (σ={sigma})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    # --- FILTROS LINEALES ---
    def filtro_promediador(self):
        """Filtro promediador"""
        if self.imagen_actual is not None:
            ksize = simpledialog.askinteger("Filtro Promediador", 
                                          "Ingrese el tamaño del kernel (3, 5, 7):",
                                          minvalue=3, maxvalue=15, initialvalue=5)
            if ksize is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = rf.filtro_promediador(img_input, ksize)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Filtro Promediador aplicado (kernel={ksize}x{ksize})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def filtro_prom_pesado(self):
        """Filtro promediador pesado"""
        if self.imagen_actual is not None:
            ksize = simpledialog.askinteger("Filtro Promediador Pesado", 
                                          "Ingrese el tamaño del kernel (3, 5, 7):",
                                          minvalue=3, maxvalue=15, initialvalue=5)
            if ksize is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = rf.filtro_promediador_pesado(img_input, ksize)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Filtro Promediador Pesado aplicado (kernel={ksize}x{ksize})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def filtro_gaussiano(self):
        """Filtro gaussiano"""
        if self.imagen_actual is not None:
            ksize = simpledialog.askinteger("Filtro Gaussiano", 
                                          "Ingrese el tamaño del kernel (3, 5, 7):",
                                          minvalue=3, maxvalue=15, initialvalue=5)
            if ksize is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = rf.filtro_gaussiano(img_input, ksize)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Filtro Gaussiano aplicado (kernel={ksize}x{ksize})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def filtro_laplaciano(self):
        """Filtro laplaciano"""
        if self.imagen_actual is not None:
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = rf.filtro_laplaciano(img_input)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info("Filtro Laplaciano aplicado")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    # --- FILTROS NO LINEALES ---
    def filtro_mediana(self):
        """Filtro de mediana"""
        if self.imagen_actual is not None:
            ksize = simpledialog.askinteger("Filtro Mediana", 
                                          "Ingrese el tamaño del kernel (3, 5, 7):",
                                          minvalue=3, maxvalue=15, initialvalue=5)
            if ksize is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = rf.filtro_mediana(img_input, ksize)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Filtro Mediana aplicado (kernel={ksize}x{ksize})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def filtro_moda(self):
        """Filtro de moda"""
        if self.imagen_actual is not None:
            ksize = simpledialog.askinteger("Filtro Moda", 
                                          "Ingrese el tamaño del kernel (3, 5, 7):",
                                          minvalue=3, maxvalue=9, initialvalue=3)
            if ksize is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = rf.filtro_moda(img_input, ksize)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Filtro Moda aplicado (kernel={ksize}x{ksize})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    
    def filtro_maximo(self):
        if self.imagen_actual is not None:
            ksize = simpledialog.askinteger(
                "Filtro Máximo",
                "Ingrese el tamaño del kernel (3, 5, 7):",
                minvalue=3, maxvalue=15, initialvalue=3
            )
            if ksize is None:
                return

            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual

            resultado = rf.filtro_maximo(img_input, ksize)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Filtro Máximo aplicado (kernel={ksize}x{ksize})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")

    
    def filtro_minimo(self):
        """Filtro de mínimo"""
        if self.imagen_actual is not None:
            ksize = simpledialog.askinteger("Filtro Mínimo", 
                                          "Ingrese el tamaño del kernel (3, 5, 7):",
                                          minvalue=3, maxvalue=15, initialvalue=3)
            if ksize is None:
                return
            
            if len(self.imagen_actual.shape) == 3:
                img_input = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
            else:
                img_input = self.imagen_actual
            
            resultado = rf.filtro_minimo(img_input, ksize)
            self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)
            self.actualizar_info(f"Filtro Mínimo aplicado (kernel={ksize}x{ksize})")
            self.mostrar_imagenes()
        else:
            messagebox.showwarning("⚠️ Advertencia", "Debe cargar una imagen primero")
    def filtro_bilateral(self):
        if self.imagen_actual is None:
            return
        gris = cv2.cvtColor(self.imagen_actual, cv2.COLOR_BGR2GRAY)
        resultado = rf.filtro_bilateral(gris)
        self.imagen_actual = cv2.cvtColor(resultado, cv2.COLOR_GRAY2BGR)  # Actualizar self.imagen_actual
        self.mostrar_imagenes()  # Llamar sin argumentos
        

        # =========================
        # MORFOLOGIA
        # =========================
    def es_binaria(self, imagen):
        if len(imagen.shape) != 2:
            return False
        valores = np.unique(imagen)
        return np.array_equal(valores, [0, 255]) or np.array_equal(valores, [0]) or np.array_equal(valores, [255])
    
if __name__ == "__main__":
    root = tk.Tk()
    app = Aplicacion(root)
    root.mainloop()