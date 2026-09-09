import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Header } from "@/components/Header";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Medicine Prescription Engine",
  description:
    "Evidence-grounded medication decision support: symptom analysis with citations traced to their original sources. Research use only.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    // Light-only by design - a clinical tool should look the same everywhere.
    <html lang="en" className={`${inter.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">
        <Header />
        <main className="flex min-h-0 flex-1 flex-col">{children}</main>
      </body>
    </html>
  );
}
