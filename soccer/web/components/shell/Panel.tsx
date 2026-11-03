import { cn } from "@/lib/format";

export function Panel({
  title, action, children, className, bodyClassName,
}: {
  title?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={cn("panel flex flex-col", className)}>
      {(title || action) && (
        <header className="flex items-center gap-3 px-4 pt-3.5 pb-2.5">
          {typeof title === "string" ? (
            <h2 className="font-display text-[13px] font-semibold tracking-tight">{title}</h2>
          ) : (
            title
          )}
          {action && <div className="ml-auto">{action}</div>}
        </header>
      )}
      <div className={cn("flex-1 min-h-0 px-4 pb-4", bodyClassName)}>{children}</div>
    </section>
  );
}
