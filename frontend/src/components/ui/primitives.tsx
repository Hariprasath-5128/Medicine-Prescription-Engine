"use client";

import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "@/lib/utils";

const buttonStyles = cva(
  "inline-flex items-center justify-center gap-2 rounded-[10px] font-medium " +
    "transition-colors outline-none focus-visible:ring-2 focus-visible:ring-offset-2 " +
    "focus-visible:ring-[var(--accent)] disabled:pointer-events-none disabled:opacity-45",
  {
    variants: {
      variant: {
        primary:
          "bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] shadow-[var(--shadow-sm)]",
        secondary:
          "bg-[var(--surface)] text-[var(--text)] border border-[var(--border)] hover:bg-[var(--surface-muted)]",
        ghost: "text-[var(--text-muted)] hover:bg-[var(--surface-muted)] hover:text-[var(--text)]",
        danger: "bg-[var(--danger)] text-white hover:brightness-95",
      },
      size: {
        sm: "h-8 px-3 text-[13px]",
        md: "h-10 px-4 text-sm",
        lg: "h-11 px-5 text-[15px]",
        icon: "h-8 w-8",
      },
    },
    defaultVariants: { variant: "primary", size: "md" },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonStyles> {}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button ref={ref} className={cn(buttonStyles({ variant, size }), className)} {...props} />
  ),
);
Button.displayName = "Button";

const badgeStyles = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
  {
    variants: {
      tone: {
        neutral: "bg-[var(--surface-muted)] text-[var(--text-muted)] border-[var(--border)]",
        accent: "bg-[var(--accent-soft)] text-[var(--accent-hover)] border-[var(--accent-border)]",
        success: "bg-[var(--success-soft)] text-[var(--success)] border-[#a7f3d0]",
        warning: "bg-[var(--warning-soft)] text-[var(--warning)] border-[var(--warning-border)]",
        danger: "bg-[var(--danger-soft)] text-[var(--danger)] border-[var(--danger-border)]",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export function Badge({
  className,
  tone,
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeStyles>) {
  return <span className={cn(badgeStyles({ tone }), className)} {...props} />;
}

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-xl border border-[var(--border)] bg-[var(--surface)] shadow-[var(--shadow-sm)]",
        className,
      )}
      {...props}
    />
  );
}

/** Horizontal score bar. `tone` shifts hue so weak scores read as weak. */
export function Meter({ value, tone = "accent" }: { value: number; tone?: "accent" | "muted" }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      className="h-1.5 w-full overflow-hidden rounded-full bg-[var(--surface-muted)]"
      role="meter"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className={cn(
          "h-full rounded-full transition-[width] duration-500 ease-out",
          tone === "accent" ? "bg-[var(--accent)]" : "bg-[var(--border-strong)]",
        )}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={cn("h-4 w-4 animate-spin", className)}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
      <path
        className="opacity-90"
        d="M12 2a10 10 0 0 1 10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}
