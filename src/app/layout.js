import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "IntelliLearn ML | Interactive Machine Learning Playground",
  description: "A serverless, client-side ML educational platform for real-time visualization and parameter tuning.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
